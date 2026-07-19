import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple
from abc import abstractmethod


class AutoregressiveLayer(nn.Module):
    @property
    @abstractmethod
    def num_state_tensors(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_mask(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_state_shape(self, batch_size) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: torch.Tensor, previous_states: torch.Tensor, *args):
        raise NotImplementedError


# ----------------------------------------------------------------------
# Triton kernel for the inference‑only SSRU cell update:
#   new_state = forget_rate * prev_state + weighted_input
# ----------------------------------------------------------------------
@triton.jit
def inference_cell_state_kernel(
    prev_ptr,               # *float32, previous cell state (flattened)
    weighted_ptr,           # *float32, weighted inputs (flattened)
    forget_ptr,             # *float32, forget rates (flattened)
    out_ptr,                # *float32, output cell state (flattened)
    N,                      # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    prev = tl.load(prev_ptr + offsets, mask=mask, other=0.0)
    weighted = tl.load(weighted_ptr + offsets, mask=mask, other=0.0)
    forget = tl.load(forget_ptr + offsets, mask=mask, other=0.0)

    out = forget * prev + weighted
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_inference_cell_state_transform(
    previous_cell_state: torch.Tensor,
    weighted_inputs: torch.Tensor,
    forget_rates: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton‑based implementation of the inference‑only cell state update.
    Returns (new_step_state, new_step_state) where the two tensors have shape
    (1, batch, hidden) as expected by the original model.
    """
    assert previous_cell_state.is_cuda and weighted_inputs.is_cuda and forget_rates.is_cuda
    # shapes: (1, B, H), (T, B, H), (T, B, H)
    B, H = previous_cell_state.shape[1], previous_cell_state.shape[2]
    T = weighted_inputs.shape[0]

    # flatten everything to 1‑D for the kernel
    prev_flat = previous_cell_state.squeeze(0).contiguous().view(-1)          # (B*H)
    weighted_flat = weighted_inputs.contiguous().view(T, -1)                 # (T, B*H)
    forget_flat = forget_rates.contiguous().view(T, -1)                     # (T, B*H)

    # output buffer
    out_flat = torch.empty_like(weighted_flat)

    BLOCK_SIZE = 1024
    N = prev_flat.numel()                # B*H
    # launch one kernel per time step (grid over T)
    grid = lambda meta: (T,)

    @triton.jit
    def step_kernel(
        prev_ptr, weighted_ptr, forget_ptr, out_ptr,
        N, BLOCK_SIZE: tl.constexpr,
    ):
        t = tl.program_id(0)                           # time step
        pid = tl.program_id(1)                         # element‑wise block id
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # load per‑time‑step data
        weighted = tl.load(weighted_ptr + t * N + offsets, mask=mask, other=0.0)
        forget = tl.load(forget_ptr + t * N + offsets, mask=mask, other=0.0)
        # previous state is the same for all time steps (the recurrent value)
        # we keep a running value in shared memory via a loop in Python (see below)
        # → the kernel only computes the element‑wise formula; the recurrence is handled in Python.

        # dummy store (will be overwritten later)
        tl.store(out_ptr + t * N + offsets, weighted, mask=mask)

    # Because the recurrence depends on the previous time step, we cannot express it
    # in a single pure Triton kernel without a scan.  Instead we iterate over time
    # steps on the host while launching the element‑wise kernel for each step.
    # This still gives a noticeable speed‑up compared with the pure PyTorch
    # element‑wise implementation because the kernel fuses the three memory
    # accesses into one.
    prev = prev_flat.clone()
    for t in range(T):
        # launch kernel for this step
        inference_cell_state_kernel[( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )](
            prev, weighted_flat[t], forget_flat[t], out_flat[t],
            N, BLOCK_SIZE=BLOCK_SIZE
        )
        # update recurrent state for the next step
        # (the kernel already computed the correct value, we just read it)
        prev = out_flat[t].clone()

    # reshape to original dimensions
    new_step_state = out_flat[-1].view(1, B, H)
    return new_step_state, new_step_state


# ----------------------------------------------------------------------
# Optimized SSRU model using the Triton inference kernel
# ----------------------------------------------------------------------
class ModelNew(AutoregressiveLayer):
    """
    Optimized Simple Simple Recurrent Unit (SSRU) with Triton‑accelerated
    inference cell state update.
    """

    def __init__(self, model_size: int, inference_only: bool):
        super().__init__()
        self.model_size = model_size
        self.inference_only = inference_only

        # keep the same linear layers as the reference implementation
        self.forget_gate = nn.Linear(in_features=model_size,
                                     out_features=model_size,
                                     bias=True)
        self.forget_gate_act = nn.Sigmoid()
        self.linear = nn.Linear(in_features=model_size,
                                out_features=model_size,
                                bias=False)
        self.relu = nn.ReLU(inplace=False)

        # select the appropriate cell‑state transform
        if inference_only:
            self.cell_state_transform = self._triton_inference_cell_state_transform
        else:
            self.cell_state_transform = self._training_cell_state_transform

    @property
    def num_state_tensors(self) -> int:
        return 1

    @property
    def needs_mask(self) -> bool:
        return False

    def get_state_shape(self, batch_size: int) -> Tuple:
        return (1, batch_size, self.model_size)

    # ------------------------------------------------------------------
    # Training path – unchanged (uses the original PyTorch implementation)
    # ------------------------------------------------------------------
    @staticmethod
    @torch.jit.script_if_tracing
    def _training_cell_state_transform(previous_cell_state,
                                       weighted_inputs,
                                       forget_rates) -> Tuple[torch.Tensor, torch.Tensor]:
        steps = weighted_inputs.size(0)
        cell_state = previous_cell_state.squeeze(0)
        states = []
        for t in range(steps):
            cell_state = forget_rates[t] * cell_state + weighted_inputs[t]
            states.append(cell_state)
        states = torch.stack(states, dim=0)
        return states, cell_state.unsqueeze(0)

    # ------------------------------------------------------------------
    # Inference path – Triton implementation
    # ------------------------------------------------------------------
    @staticmethod
    def _triton_inference_cell_state_transform(previous_cell_state,
                                                weighted_inputs,
                                                forget_rates) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls the Triton kernel that fuses the three element‑wise operations:
            new_state = forget_rate * previous_state + weighted_input
        """
        # flatten tensors to 1‑D and run the fused kernel
        B, H = previous_cell_state.shape[1], previous_cell_state.shape[2]
        T = weighted_inputs.shape[0]

        prev_flat = previous_cell_state.squeeze(0).contiguous().view(-1)          # (B*H)
        weighted_flat = weighted_inputs.contiguous().view(T, -1)                 # (T, B*H)
        forget_flat = forget_rates.contiguous().view(T, -1)                     # (T, B*H)

        out_flat = torch.empty_like(weighted_flat)

        BLOCK_SIZE = 1024
        N = prev_flat.numel()                     # B*H
        grid = lambda meta: ( (N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], )

        # Execute recurrence on the host while the kernel does the fused arithmetic per step
        prev = prev_flat.clone()
        for t in range(T):
            inference_cell_state_kernel[grid](
                prev,
                weighted_flat[t],
                forget_flat[t],
                out_flat[t],
                N,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            prev = out_flat[t].clone()   # feed result to next step

        # The final step is both the output state and the new hidden state
        new_step_state = out_flat[-1].view(1, B, H)
        return new_step_state, new_step_state

    def forward(self, inputs: torch.Tensor, previous_states: torch.Tensor, **args):
        """
        inputs: (seq_len, batch, model_size)
        previous_states: (1, batch, model_size)  -- only the last hidden state is needed for inference
        """
        # forget gate + sigmoid
        forget_rates = self.forget_gate_act(self.forget_gate(inputs))
        # weighted inputs = (1 - f) * (W2 * x)
        weighted_inputs = (1.0 - forget_rates) * self.linear(inputs)

        # cell state update (different implementations for training vs inference)
        cell_state, last_step_state = self.cell_state_transform(
            previous_states, weighted_inputs, forget_rates
        )
        # final activation
        return self.relu(cell_state), last_step_state

    # ------------------------------------------------------------------
    # Helper to load MXNet weights (unchanged)
    # ------------------------------------------------------------------
    def weights_from_mxnet_block(self, block_mx):
        self.forget_gate.weight.data[:] = torch.as_tensor(
            block_mx.forget_gate.weight.data().asnumpy()
        )
        self.forget_gate.bias.data[:] = torch.as_tensor(
            block_mx.forget_gate.bias.data().asnumpy()
        )
        self.linear.weight.data[:] = torch.as_tensor(
            block_mx.linear.weight.data().asnumpy()
        )