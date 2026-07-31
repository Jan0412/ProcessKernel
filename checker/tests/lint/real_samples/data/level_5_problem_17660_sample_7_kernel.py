import torch
import torch.nn as nn
import triton
import triton.language as tl
import numpy as np
from typing import Iterable, Optional


# ------------------- Triton kernels ------------------- #
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    sigmoid_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


@triton.jit
def ge_kernel(x_ptr, out_ptr, n_elements, threshold, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x >= threshold, 1, 0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_ge(x: torch.Tensor, threshold: float) -> torch.Tensor:
    assert x.is_cuda
    x = x.contiguous()
    out = torch.empty_like(x, dtype=torch.long)
    n = x.numel()
    BLOCK = 128
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    ge_kernel[grid](x, out, n, threshold, BLOCK_SIZE=BLOCK)
    return out


# ------------------- Helper functions (unchanged) ------------------- #
def find_closest_span_pairs(head: 'Iterable', tail: 'Iterable', backtrace: 'Optional[bool]' = True):
    if isinstance(head, torch.Tensor):
        head = head.detach().cpu()
    if isinstance(tail, torch.Tensor):
        tail = tail.detach().cpu()
    head_valid_poses = np.where(head == 1)[0]
    tail_valid_poses = np.where(tail == 1)[0]
    tail_used_poses = {pos: (False) for pos in tail_valid_poses.tolist()}
    pairs = []
    for head_i in head_valid_poses:
        tail_js = tail_valid_poses[tail_valid_poses >= head_i]
        if len(tail_js) > 0:
            tail_j = tail_js[0]
            tail_used_poses[tail_j] = True
            pairs.append((head_i, tail_j))
    if backtrace:
        for tail_j in tail_used_poses:
            if not tail_used_poses[tail_j]:
                head_is = head_valid_poses[head_valid_poses <= tail_j]
                if len(head_is) > 0:
                    head_i = head_is[-1]
                    pairs.append((head_i, tail_j))
    return pairs


def find_closest_span_pairs_with_index(heads: 'Iterable', tails: 'Iterable',
                                      backtrace: 'Optional[bool]' = True):
    results = []
    for idx, (head, tail) in enumerate(zip(heads, tails)):
        pairs = find_closest_span_pairs(head, tail, backtrace=backtrace)
        for pair in pairs:
            results.append((idx, pair[0], pair[1]))
    return results


# ------------------- Optimized Model ------------------- #
class ModelNew(nn.Module):
    """
    Optimized version of SubjObjSpan using Triton kernels for
    element‑wise addition, sigmoid, and binary thresholding.
    """

    def __init__(self, hidden_size, num_classes, threshold: Optional[float] = 0.5):
        super().__init__()
        self.threshold = threshold
        self.subj_head_ffnn = nn.Linear(hidden_size, 1)
        self.subj_tail_ffnn = nn.Linear(hidden_size, 1)
        self.obj_head_ffnn = nn.Linear(hidden_size, num_classes)
        self.obj_tail_ffnn = nn.Linear(hidden_size, num_classes)

    # -----------------------------------------------------------------
    # Helper that builds the mapping tensors (unchanged)
    # -----------------------------------------------------------------
    def build_batch_mapping(self, subj_head, subj_tail):
        subjs = find_closest_span_pairs(subj_head, subj_tail)
        seq_len = subj_head.shape[0]
        if len(subjs) > 0:
            subjs_head_mapping = torch.zeros(len(subjs), seq_len, device=subj_head.device)
            subjs_tail_mapping = torch.zeros(len(subjs), seq_len, device=subj_tail.device)
            for subj_idx, subj in enumerate(subjs):
                subjs_head_mapping[subj_idx, subj[0]] = 1.0
                subjs_tail_mapping[subj_idx, subj[1]] = 1.0
            return subjs, subjs_head_mapping, subjs_tail_mapping
        else:
            return None, None, None

    # -----------------------------------------------------------------
    # Core computation – unchanged except for the addition step
    # -----------------------------------------------------------------
    def get_objs_for_specific_subj(self, subj_head_mapping, subj_tail_mapping, hidden):
        # weighted sums (matmul) – keep PyTorch for simplicity/accuracy
        subj_head = torch.matmul(subj_head_mapping, hidden)          # (B, H)
        subj_tail = torch.matmul(subj_tail_mapping, hidden)          # (B, H)
        sub = (subj_head + subj_tail) * 0.5                         # (B, H)

        # Broadcast sub to (B, L, H) and add to hidden using Triton
        sub_exp = sub.unsqueeze(1)                                   # (B,1,H)
        encoded_text = triton_add(hidden, sub_exp)                  # (B,L,H)

        pred_obj_heads = self.obj_head_ffnn(encoded_text)           # (B,L,Cls)
        pred_obj_tails = self.obj_tail_ffnn(encoded_text)           # (B,L,Cls)
        return pred_obj_heads, pred_obj_tails

    # -----------------------------------------------------------------
    # Forward (training) – keep original behaviour (no Triton needed)
    # -----------------------------------------------------------------
    def forward(self, hidden, subj_head, subj_tail):
        subj_head_out = self.subj_head_ffnn(hidden)
        subj_tail_out = self.subj_tail_ffnn(hidden)
        obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(
            subj_head.unsqueeze(1), subj_tail.unsqueeze(1), hidden)
        return subj_head_out.squeeze(-1), subj_tail_out.squeeze(-1), obj_head_out, obj_tail_out

    # -----------------------------------------------------------------
    # Inference – fully Triton‑accelerated sigmoid & threshold logic
    # -----------------------------------------------------------------
    def predict(self, hidden):
        if hidden.shape[0] != 1:
            raise RuntimeError(
                f'eval batch size must be 1 x hidden_size, while hidden is {hidden.shape}'
            )
        # ---- subject head / tail logits ----
        subj_head_out = self.subj_head_ffnn(hidden)          # (1, L, 1)
        subj_tail_out = self.subj_tail_ffnn(hidden)          # (1, L, 1)

        # ---- Triton sigmoid ----
        subj_head_out = triton_sigmoid(subj_head_out)
        subj_tail_out = triton_sigmoid(subj_tail_out)

        # ---- Threshold (>=) ----
        pred_subj_head = triton_ge(subj_head_out, self.threshold).long()
        pred_subj_tail = triton_ge(subj_tail_out, self.threshold).long()

        # ---- Build mappings for the (single) batch ----
        subjs, subj_head_mappings, subj_tail_mappings = self.build_batch_mapping(
            pred_subj_head.squeeze(0).squeeze(-1),
            pred_subj_tail.squeeze(0).squeeze(-1)
        )

        triples = []
        if subjs is not None:
            # ---- Object predictions for each subject ----
            obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(
                subj_head_mappings.unsqueeze(1),
                subj_tail_mappings.unsqueeze(1),
                hidden
            )
            # ---- Triton sigmoid + threshold for objects ----
            obj_head_out = triton_sigmoid(obj_head_out)
            obj_tail_out = triton_sigmoid(obj_tail_out)

            obj_head_out = triton_ge(obj_head_out, self.threshold).long()
            obj_tail_out = triton_ge(obj_tail_out, self.threshold).long()

            # ---- Decode spans ----
            for subj_idx, subj in enumerate(subjs):
                # permute to (Cls, L) as original code expects
                objs = find_closest_span_pairs_with_index(
                    obj_head_out[subj_idx].permute(1, 0),
                    obj_tail_out[subj_idx].permute(1, 0)
                )
                for relation_idx, obj_pair_start, obj_pair_end in objs:
                    triples.append(((subj[0], subj[1] + 1), relation_idx,
                                    (obj_pair_start, obj_pair_end + 1)))
        return [triples]


# ------------------- Compatibility shim ------------------- #
# The original adapter expects a class named Model; expose ModelNew under that name.
Model = ModelNew