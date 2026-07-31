import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm


# --------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------

# Linear (input_dim -> 1) + sigmoid fused
@triton.jit
def linear_sigmoid_kernel(
    x_ptr,          # [N, T, D] input
    w_ptr,          # [D] weight
    b_ptr,          # [] bias
    out_ptr,        # [N, T] output (sigmoid)
    N, T, D,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # map each program to a (n, t) pair
    n = pid // T
    t = pid % T
    # base offset for this row
    row_offset = n * T * D + t * D

    acc = tl.float32(0.0)
    for d in range(0, D, BLOCK_D):
        cur_off = row_offset + d
        mask = (d + tl.arange(0, BLOCK_D)) < D
        x = tl.load(x_ptr + cur_off + tl.arange(0, BLOCK_D), mask=mask, other=0.0)
        w = tl.load(w_ptr + d + tl.arange(0, BLOCK_D), mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)

    # add bias
    b = tl.load(b_ptr)
    acc = acc + b
    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-acc))
    # store
    tl.store(out_ptr + n * T + t, out)


# Mean aggregation over time dimension
@triton.jit
def mean_agg_kernel(
    pi_ptr,          # [N, T]
    out_ptr,         # [N]
    N, T,
    BLOCK_T: tl.constexpr,
):
    n = tl.program_id(0)
    sum_val = tl.float32(0.0)
    for off in range(0, T, BLOCK_T):
        cur_off = n * T + off
        mask = (off + tl.arange(0, BLOCK_T)) < T
        vals = tl.load(pi_ptr + cur_off + tl.arange(0, BLOCK_T), mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
    mean_val = sum_val / T
    tl.store(out_ptr + n, mean_val)


# Max aggregation over time dimension
@triton.jit
def max_agg_kernel(
    pi_ptr,          # [N, T]
    out_ptr,         # [N]
    N, T,
    BLOCK_T: tl.constexpr,
):
    n = tl.program_id(0)
    max_val = tl.full([1], -float("inf"), dtype=tl.float32)[0]
    for off in range(0, T, BLOCK_T):
        cur_off = n * T + off
        mask = (off + tl.arange(0, BLOCK_T)) < T
        vals = tl.load(pi_ptr + cur_off + tl.arange(0, BLOCK_T), mask=mask, other=-float("inf"))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))
    tl.store(out_ptr + n, max_val)


# Cross‑time‑steps loss (mean of squared diff)
@triton.jit
def cross_time_loss_kernel(
    pi_ptr,          # [N, T]
    out_ptr,         # [] scalar loss
    N, T,
    BLOCK_T: tl.constexpr,
):
    sum_sq = tl.float32(0.0)
    total = tl.int32(0)
    for off in range(0, T - 1, BLOCK_T):
        cur_off = off
        mask = (off + tl.arange(0, BLOCK_T)) < (T - 1)
        # pi[:, off] and pi[:, off+1]
        a = tl.load(pi_ptr + tl.arange(0, N) * T + cur_off + tl.arange(0, BLOCK_T)[None, :], mask=mask[None, :], other=0.0)
        b = tl.load(pi_ptr + tl.arange(0, N) * T + cur_off + tl.arange(0, BLOCK_T)[None, :] + 1, mask=mask[None, :], other=0.0)
        diff = a - b
        sum_sq += tl.sum(diff * diff, axis=[0, 1])
        total += tl.sum(mask, axis=0) * N
    loss = sum_sq / total.to(tl.float32)
    tl.store(out_ptr, loss)


# --------------------------------------------------------------
# Helper wrappers
# --------------------------------------------------------------

def triton_linear_sigmoid(x, weight, bias):
    assert x.is_cuda and weight.is_cuda and bias.is_cuda
    N, T, D = x.shape
    out = torch.empty((N, T), dtype=x.dtype, device=x.device)

    BLOCK_D = 64
    grid = (N * T,)
    linear_sigmoid_kernel[grid](
        x, weight, bias, out,
        N, T, D,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out


def triton_mean_agg(pi):
    N, T = pi.shape
    out = torch.empty((N,), dtype=pi.dtype, device=pi.device)
    BLOCK_T = 64
    grid = (N,)
    mean_agg_kernel[grid](
        pi, out,
        N, T,
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out


def triton_max_agg(pi):
    N, T = pi.shape
    out = torch.empty((N,), dtype=pi.dtype, device=pi.device)
    BLOCK_T = 64
    grid = (N,)
    max_agg_kernel[grid](
        pi, out,
        N, T,
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out


def triton_cross_time_loss(pi):
    N, T = pi.shape
    out = torch.empty((1,), dtype=pi.dtype, device=pi.device)
    BLOCK_T = 64
    grid = (1,)
    cross_time_loss_kernel[grid](
        pi, out,
        N, T,
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out[0]


# --------------------------------------------------------------
# Optimized model
# --------------------------------------------------------------

class ModelNew(nn.Module):
    def __init__(self, input_dim, flight_length, device,
                 aggregation='maxpool', output_resize=False):
        super().__init__()
        self.input_dim = input_dim
        self.flight_length = flight_length
        self.device = device
        self.agg = aggregation
        self.output_resize = output_resize

        # Linear weight & bias (input_dim -> 1)
        self.weight = nn.Parameter(torch.randn(input_dim, dtype=torch.float32, device=device))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32, device=device))

    def forward(self, x, train=True):
        """
        x : [N, flight_length, input_dim]   (torch.cuda.FloatTensor)
        Returns p : [N, 1]
        """
        if not x.is_cuda:
            # fallback to PyTorch ops for CPU
            pi = torch.sigmoid(F.linear(x, self.weight.unsqueeze(0), self.bias)).squeeze(-1)
        else:
            pi = triton_linear_sigmoid(x, self.weight, self.bias)

        self.pi = pi  # store for loss computation

        if self.agg == 'mean':
            p = triton_mean_agg(pi) if pi.is_cuda else pi.mean(dim=-1)
        else:  # maxpool
            p = triton_max_agg(pi) if pi.is_cuda else pi.max(dim=-1).values

        return p.view(-1, 1)

    # ------------------------------------------------------------------
    # The rest of the methods are unchanged except that cross_time_steps_loss
    # now uses the Triton implementation when possible.
    # ------------------------------------------------------------------
    def get_feature_importance(self, columns, n_top=5):
        coeffs = self.weight.detach().cpu().numpy().flatten()
        sorted_feat_idx = np.argsort(coeffs)[::-1]
        sorted_columns = columns[sorted_feat_idx[:n_top]]
        top_values = coeffs[sorted_feat_idx[:n_top]]
        return sorted_columns, top_values

    def cross_time_steps_loss(self, Pi):
        if Pi.is_cuda:
            return triton_cross_time_loss(Pi)
        else:
            diff = (Pi[:, :-1] - Pi[:, 1:]) ** 2
            return torch.mean(torch.mean(diff, dim=-1))

    # ------------------------------------------------------------------
    # Training routine (kept as‑is, only minor device handling changes)
    # ------------------------------------------------------------------
    def train_LR(self, X_train, y_train, X_val, y_val, batch_size,
                 print_every_epochs=5, l2=0.001, learning_rate=0.001,
                 use_stratified_batch_size=True, verbose=1, num_epochs=100,
                 optimizer='adam', momentum=0.99):
        self.train()
        if isinstance(self.device, str) and 'cuda' in self.device:
            self.cuda()
        else:
            self.cpu()

        criterion = nn.BCELoss()
        if optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=learning_rate,
                                   weight_decay=l2)
        else:
            opt = torch.optim.SGD(self.parameters(), lr=learning_rate,
                                  momentum=momentum, weight_decay=l2)

        # tensors
        if not torch.is_tensor(X_train):
            X_train = torch.tensor(X_train, device=self.device, dtype=torch.float32)
        if not torch.is_tensor(y_train):
            y_train = torch.tensor(y_train, device=self.device, dtype=torch.float32).view(-1)

        if X_val is not None:
            if not torch.is_tensor(X_val):
                X_val = torch.tensor(X_val, device=self.device, dtype=torch.float32)
            if not torch.is_tensor(y_val):
                y_val = torch.tensor(y_val, device=self.device, dtype=torch.float32)

        train_dataset = myDataset(X_train, y_train)
        if use_stratified_batch_size:
            # compute class weights
            uniq = torch.unique(y_train)
            weights = []
            for lbl in uniq:
                cnt = (y_train == lbl).sum().item()
                weights.append(1.0 / cnt)
            class_weights = torch.tensor(weights, device=self.device)
            sample_weights = class_weights[y_train.long()]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
            loader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = myDataset(X_val, y_val)
            loader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # metrics containers
        hist = np.zeros(num_epochs)
        val_hist = np.zeros(num_epochs)
        b_acc = np.zeros(num_epochs)
        val_b_acc = np.zeros(num_epochs)
        f1 = np.zeros(num_epochs)
        val_f1 = np.zeros(num_epochs)

        try:
            for epoch in tqdm(range(num_epochs)):
                batch_acc, batch_val_acc = [], []
                batch_f1, batch_val_f1 = [], []

                for batch_x, batch_y in loader_train:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                    outputs = self.forward(batch_x)                # [N,1]
                    g_loss = self.cross_time_steps_loss(self.pi)   # scalar
                    loss = criterion(outputs.squeeze(), batch_y) + g_loss

                    hist[epoch] = loss.item()

                    # metrics
                    pred = (outputs.cpu().detach().numpy() > self.threshold).astype(int).squeeze()
                    true = batch_y.cpu().detach().numpy()
                    b_acc[epoch] = balanced_accuracy_score(true, pred)
                    batch_acc.append(b_acc[epoch])
                    batch_f1.append(f1_score(true, pred, average='binary'))

                    # backward
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # validation
                if X_val is not None:
                    val_losses = []
                    for vx, vy in loader_val:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        with torch.no_grad():
                            vout = self.forward(vx)
                            v_g = self.cross_time_steps_loss(self.pi)
                            v_loss = criterion(vout.squeeze(), vy) + v_g
                            val_losses.append(v_loss.item())

                            v_pred = (vout.cpu().detach().numpy() > self.threshold).astype(int).squeeze()
                            v_true = vy.cpu().detach().numpy()
                            val_b_acc[epoch] = balanced_accuracy_score(v_true, v_pred)
                            batch_val_acc.append(val_b_acc[epoch])
                            batch_val_f1.append(f1_score(v_true, v_pred, average='binary'))
                    val_hist[epoch] = np.mean(val_losses)

                if verbose and epoch % print_every_epochs == 0:
                    print(f"Epoch {epoch:3d} | loss {hist[epoch]:.4f} | val loss {val_hist[epoch]:.4f}")

                # aggregate epoch metrics
                b_acc[epoch] = np.mean(batch_acc)
                f1[epoch] = np.mean(batch_f1)
                if X_val is not None:
                    val_b_acc[epoch] = np.mean(batch_val_acc)
                    val_f1[epoch] = np.mean(batch_val_f1)

        except KeyboardInterrupt:
            pass

        # store for later inspection
        self.hist = hist
        self.val_hist = val_hist
        self.b_acc = b_acc
        self.val_b_acc = val_b_acc
        self.f1 = f1
        self.val_f1 = val_f1
        self.eval()
        return self

    def fit(self, **kw):
        return self.train_LR(**kw)


# --------------------------------------------------------------
# Compatibility shim (same signature as original adapter)
# --------------------------------------------------------------
def get_inputs():
    return [torch.rand([4, 4, 4], device='cuda')]

def get_init_inputs():
    # input_dim, flight_length, device
    return [4, 4, 'cuda']

# Model alias for the benchmark harness
Model = ModelNew