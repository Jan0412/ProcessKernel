@triton.jit
def hinge_loss_kernel(
    predictions_ptr,
    targets_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    preds = tl.load(predictions_ptr + offsets, mask=mask, other=0.0)
    tgts = tl.load(targets_ptr + offsets, mask=mask, other=0.0)

    # Hinge loss: max(0, 1 - pred * target)
    val = 1.0 - preds * tgts
    clamped = tl.maximum(val, 0.0)

    # Block-local reduction
    block_sum = tl.sum(clamped, axis=0)

    # Global reduction via atomic add
    tl.atomic_add(out_ptr, block_sum)