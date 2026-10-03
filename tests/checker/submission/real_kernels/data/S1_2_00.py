import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softplus_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting index for this program
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundaries
    mask = offsets < n_elements
    
    # Load input values
    # We use 0.0 as the default value for out-of-bounds, though mask prevents usage
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute softplus: log(1 + exp(x))
    # Numerically stable implementation:
    # For x > 0: x + log(1 + exp(-x))
    # For x <= 0: log(1 + exp(x))
    
    # We can use tl.where to branch
    # But simpler logic without branching for vectorization?
    # Triton handles branching well, but let's try to be clean.
    
    # Option 1: Branching
    # is_pos = x > 0
    # val_pos = x + tl.log1p(tl.exp(-x))
    # val_neg = tl.log1p(tl.exp(x))
    # out = tl.where(is_pos, val_pos, val_neg)
    
    # Option 2: Unified formula?
    # log(1 + exp(x)) is hard to do unified without overflow risk in exp(x).
    # But we can clamp x? No.
    # The branching approach is standard and safe.
    
    is_pos = x > 0.0
    # For positive x, exp(-x) is safe (<= 1)
    # For negative x, exp(x) is safe (<= 1)
    
    # Note: tl.exp(-x) when x is large positive is 0.
    # tl.exp(x) when x is large negative is 0.
    
    # Compute terms
    # We need to compute exp(-x) and exp(x). 
    # Actually, we only need one depending on sign, but computing both is cheap?
    # No, let's stick to where.
    
    # However, tl.where evaluates both branches in Triton? 
    # Yes, usually. So we compute both exp(-x) and exp(x).
    # If x is large positive, exp(x) overflows. 
    # Wait, if we compute exp(x) inside the 'else' branch of a where, 
    # Triton might still evaluate it if not careful or if the hardware executes all lanes.
    # In Triton, `tl.where` is a selector, but inputs are computed before selection?
    # Actually, Triton compiles to PTX. If `tl.where` is used, the arguments are computed.
    # If `exp(x)` overflows, it produces Inf. `log1p(Inf)` is Inf.
    # If `x` is large positive, `is_pos` is true. We select `val_pos`. `val_neg` is Inf.
    # The result is `val_pos` (finite). So it works, provided Inf doesn't crash or corrupt.
    # Inf is a valid float. It should be fine.
    # BUT, if `x` is very large (e.g. 1000), `exp(x)` is Inf.
    # Is it safe to generate Inf? Yes.
    
    # However, to be strictly clean and avoid generating Infs if possible:
    # We can clamp x for the exp calculation?
    # Or just rely on the fact that we pick the finite result.
    
    # Let's check if we can avoid computing exp(x) for positive x.
    # We can't easily in a vectorized kernel without divergent warps, but Triton handles that.
    # Actually, `tl.where` in Triton is an instruction `Selp`. It selects between two values.
    # Both values must be ready. So both branches execute.
    # So `exp(x)` will execute for positive x.
    # If x=100, exp(100) -> Inf.
    # log1p(Inf) -> Inf.
    # Result is selected as `x + log1p(exp(-100))` = 100.
    # So the output is correct. The Inf is discarded.
    # This is acceptable.
    
    # To be safer against potential NaNs or weird hardware behavior with Inf in log1p:
    # log1p(Inf) is Inf. It's defined.
    
    # Let's implement.
    
    # Using log1p for better precision near 0
    # exp(-x) for x > 0 is in (0, 1]
    # exp(x) for x <= 0 is in (0, 1]
    
    # We can compute:
    # term1 = tl.exp(-x) # Safe for x > 0, might be huge for x < 0 (overflow)
    # term2 = tl.exp(x)  # Safe for x <= 0, might be huge for x > 0 (overflow)
    
    # Wait, if x is negative (e.g. -100), -x is 100. exp(100) overflows.
    # So computing exp(-x) for negative x causes overflow.
    # So we CANNOT compute both blindly if x range is large.
    # x range is float32, can be -1000 to 1000.
    # exp(1000) overflows.
    
    # So we MUST mask or use logic to prevent overflow in the unused branch?
    # Or use a different formula.
    
    # Formula: log(1 + exp(x))
    # If we just do log(1 + exp(x)), exp(x) overflows for x > 88.
    # We need the stable version.
    
    # Stable version logic:
    # if x > 0: x + log1p(exp(-x))
    # else: log1p(exp(x))
    
    # To implement this without overflow in the "unused" branch of `tl.where`:
    # We can clamp the input to exp?
    # Or just accept Inf?
    # If x = -100, -x = 100. exp(100) -> Inf.
    # log1p(Inf) -> Inf.
    # x + Inf -> Inf.
    # But we are in the `else` branch (x <= 0), so we pick `log1p(exp(x))`.
    # exp(-100) -> 0. log1p(0) -> 0.
    # Result 0. Correct.
    # The `Inf` from the first branch is discarded.
    # Is this safe?
    # In CUDA/PTX, operations on Inf are defined. Inf + (-100) = Inf.
    # Selp selects the second operand.
    # So it should be fine.
    
    # However, some compilers might optimize or warn.
    # Let's try to be safe.
    # We can clamp x to a safe range before passing to exp?
    # No, because we need the sign.
    
    # Alternative:
    # max_x = tl.maximum(x, 0.0)
    # min_x = tl.minimum(x, 0.0) # Not needed
    
    # Actually, PyTorch's softplus uses a threshold.
    # "For x > threshold, return x".
    # Threshold is usually 20.
    # If we clamp x to [-20, 20] or something? No.
    
    # Let's stick to the branching logic. It's standard.
    # To avoid overflow in `exp(-x)` when `x` is large negative:
    # We can check `x < -20`?
    # If `x < -20`, `exp(x)` is ~0. `log1p(0) = 0`.
    # So for `x < -20`, output is 0.
    # If `x > 20`, output is `x`.
    # Between -20 and 20, compute normally.
    
    # This avoids computing exp of large numbers entirely.
    # Let's implement this thresholding logic. It's robust and matches PyTorch behavior.
    
    # Threshold
    threshold = 20.0
    
    # Case 1: x > threshold -> output x
    # Case 2: x < -threshold -> output 0 (approx, since exp(x) ~ 0)
    # Case 3: else -> log(1 + exp(x))
    
    # Wait, softplus(-20) = log(1 + exp(-20)) ~ exp(-20) ~ 2e-9.
    # 0 is a very good approximation.
    
    # Let's refine:
    # out = tl.where(x > threshold, x, tl.log1p(tl.exp(x)))
    # But wait, if x is large positive (e.g. 100), and we don't clamp, exp(100) overflows.
    # But if x > 20, we return x. We don't compute exp(x).
    # So we need nested where or logic.
    
    # Logic:
    # 1. Check if x > 20. If so, res = x.
    # 2. Else, compute log1p(exp(x)).
    #    Inside "Else", x <= 20. exp(x) <= exp(20) ~ 4.8e8. Safe.
    #    What about negative x? exp(x) is small. Safe.
    #    So computing exp(x) is safe if x <= 20.
    
    # So:
    # is_large = x > 20.0
    # safe_exp = tl.where(is_large, 0.0, tl.exp(x)) # Just to be safe, though if is_large is true we don't use it?
    # Wait, if we use tl.where(is_large, x, log1p(exp(x))), the exp(x) is evaluated regardless?
    # Yes.
    # So if x=100, exp(100) overflows.
    # We need to prevent exp(100) from executing or overflowing.
    
    # We can clamp x before exp?
    # x_clamped = tl.minimum(x, 20.0)
    # But if x was 100, we want output 100.
    # If we clamp x to 20, compute log1p(exp(20)) ~ 20.
    # Then we can use the original x?
    # No, log1p(exp(20)) is approx 20.000...
    # We want exactly x (100).
    
    # So:
    # out = tl.where(x > 20.0, x, tl.log1p(tl.exp(x)))
    # Problem: exp(x) overflows for x=100.
    
    # Solution:
    # Compute exp only on clamped x?
    # But we need to select between x and log1p...
    
    # Maybe:
    # x_clamped = tl.minimum(x, 20.0)
    # val = tl.log1p(tl.exp(x_clamped))
    # out = tl.where(x > 20.0, x, val)
    
    # Check:
    # If x = 100. x_clamped = 20. val = log1p(exp(20)) ~ 20.
    # x > 20 is True. out = 100. Correct.
    # If x = 10. x_clamped = 10. val = log1p(exp(10)) ~ 10.000045.
    # x > 20 is False. out = val. Correct.
    # If x = -100. x_clamped = -100. val = log1p(exp(-100)) ~ 0.
    # x > 20 is False. out = val. Correct.
    
    # Is `tl.minimum` safe? Yes.
    # Is `exp(x_clamped)` safe? Max x_clamped is 20. exp(20) is safe.
    # Min x_clamped is -inf. exp(-inf) is 0. Safe.
    
    # This looks very robust.
    
    # Wait, what about x slightly above 20?
    # x = 21.
    # x_clamped = 20.
    # val = log1p(exp(20)) ~ 20.000045.
    # x > 20 is True.
    # out = 21.
    # Correct?
    # Softplus(21) = log(1 + exp(21)) = 21 + log(1 + exp(-21)) ~ 21 + 7e-10 ~ 21.
    # So returning 21 is a very good approximation.
    # PyTorch uses threshold 20 exactly for this reason.
    
    # So the strategy:
    # 1. Clamp x to max 20.0 for the exponential calculation.
    # 2. Compute log1p(exp(clamped_x)).
    # 3. If original x > 20.0, return original x. Else return computed value.
    
    # Wait, `tl.minimum` is element-wise.
    # `x_clamped = tl.minimum(x, 20.0)`
    
    # One edge case: x is NaN?
    # Softplus(NaN) should be NaN.
    # exp(NaN) -> NaN. log1p(NaN) -> NaN.
    # x > 20 -> False (usually).
    # Returns NaN. Correct.
    
    # Code:
    # x_clamped = tl.minimum(x, 20.0)
    # exp_val = tl.exp(x_clamped)
    # log_val = tl.log1p(exp_val)
    # out = tl.where(x > 20.0, x, log_val)
    
    # This seems perfect.
    
    # Wait, `tl.minimum` might not be available? It is. `tl.min` is reduction. `tl.minimum` is element-wise.
    # Actually, `tl.minimum` is available in `triton.language`.
    # Or we can use `tl.where(x > 20.0, 20.0, x)`.
    
    # Let's check imports. `triton.language` has `minimum`?
    # In recent Triton, yes.
    # If not, `tl.where(x > 20.0, 20.0, x)` works.
    
    # Let's use `tl.where` for clamping to be safe across versions.
    
    x_clamped = tl.where(x > 20.0, 20.0, x)
    exp_val = tl.exp(x_clamped)
    log_val = tl.log1p(exp_val)
    out = tl.where(x > 20.0, x, log_val)
    
    # Wait, if x > 20, x_clamped is 20. exp_val is exp(20). log_val is ~20.
    # out selects x (which is > 20).
    # So log_val is computed but discarded.
    # exp(20) is safe.
    # What if x was 1000?
    # x_clamped is 20. exp(20) safe.
    # out selects 1000.
    # Safe.
    
    # What if x is very negative?
    # x = -1000.
    # x_clamped = -1000.
    # exp(-1000) = 0.
    # log1p(0) = 0.
    # x > 20 False.
    # out = 0.
    # Correct.
    
    # This logic seems solid.

    # Wait, is `20.0` the best threshold?
    # PyTorch default is 20.
    # For float32, precision is ~1e-7.
    # exp(-20) ~ 2e-9.
    # log(1 + 2e-9) ~ 2e-9.
    # x + 2e-9 vs x.
    # Difference is 2e-9.
    # Float32 epsilon at 20 is 20 * 1e-7 = 2e-6.
    # So 2e-9 is well below machine epsilon.
    # So x is the correct representation for softplus(x) when x > 20.
    
    # One detail: `tl.log1p` might not be in older Triton.
    # `tl.log(1.0 + exp_val)` is safer compatibility-wise?
    # `log1p` is better for precision.
    # I'll use `tl.log1p` as it's standard in modern Triton (which is implied by the context of "custom Triton kernels").
    
    # Let's write the kernel.

    pass # placeholder

def triton_softplus(x: torch.Tensor):
    # ...
    pass

class ModelNew(nn.Module):
    # ...
    pass