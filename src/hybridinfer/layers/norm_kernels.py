"""Row-wise norm kernels whose reduction layout does not depend on row count."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gemma_norm(X, Residual, Weight, Output, ResidualOutput,
                SX: tl.constexpr, SR: tl.constexpr, WIDTH: tl.constexpr,
                EPS: tl.constexpr, HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < WIDTH
    x = tl.load(X+row*SX+col, mask, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(Residual+row*SR+col, mask, other=0).to(tl.float32)
        tl.store(ResidualOutput+row*WIDTH+col, x, mask)
    variance = tl.sum(x*x, 0)/WIDTH
    inverse = libdevice.rsqrt(variance+EPS)
    weight = tl.load(Weight+col, mask, other=0).to(tl.float32)
    y = (x*inverse)*(1.0+weight)
    tl.store(Output+row*WIDTH+col, y, mask)


def gemma_norm(x, weight, eps, residual=None):
    """Normalize the FP32 residual sum; store residual in the input dtype.

    A fixed reduction layout per feature width makes decode and packed rows
    use the same arithmetic. Validated against vLLM 0.19.0 static-shape
    compiled norms with precision-cast emulation disabled. Dynamic compiler
    specializations and complete-engine parity require separate validation.
    """
    width = x.shape[-1]
    flat = x.reshape(-1, width)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    summed = None
    previous = flat
    if residual is not None:
        previous = residual.reshape(-1, width)
        if previous.stride(-1) != 1:
            previous = previous.contiguous()
        summed = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _gemma_norm[(flat.shape[0],)](flat, previous, weight, output,
        output if summed is None else summed, flat.stride(0), previous.stride(0),
        width, eps, residual is not None, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=True)
    return output if summed is None else (output, summed)
