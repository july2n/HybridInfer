"""Packed causal convolution and FP32 indexed GDN decode kernels."""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv(X, W, S, Slots, Cu, Y, C: tl.constexpr, K: tl.constexpr,
          WS: tl.constexpr, WK: tl.constexpr, DECODE: tl.constexpr,
          BT: tl.constexpr, BC: tl.constexpr):
    req = tl.program_id(2)
    slot = tl.load(Slots + req)
    if DECODE:
        start, end = req, req + 1
    else:
        start, end = tl.load(Cu + req), tl.load(Cu + req + 1)
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    acc = tl.full((BT, BC), 0, tl.float32)
    for i in tl.static_range(K):
        pos = t + i - (K - 1)
        current = tl.load(X + (start + pos[:, None]) * C + c[None, :],
                          (pos[:, None] >= 0) & (t[:, None] < end - start) & (c[None, :] < C), other=0)
        history = tl.load(S + (slot * C + c[None, :]) * (K - 1) + (pos[:, None] + K - 1),
                          (pos[:, None] < 0) & (t[:, None] < end - start) & (c[None, :] < C), other=0)
        value = tl.where(pos[:, None] >= 0, current, history).to(tl.float32)
        weight = tl.load(W + c * WS + i * WK, c < C, other=0).to(tl.float32)
        acc += value * weight[None, :]
    # Preserve the BF16/FP16 conv output rounding before SiLU.
    rounded = acc.to(X.dtype.element_ty).to(tl.float32)
    result = rounded * tl.sigmoid(rounded)
    tl.store(Y + (start + t[:, None]) * C + c[None, :], result,
             (t[:, None] < end - start) & (c[None, :] < C))


@triton.jit
def _conv_state(X, S, Slots, Cu, C: tl.constexpr, WIDTH: tl.constexpr,
                DECODE: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    req = tl.program_id(1)
    slot = tl.load(Slots + req)
    if DECODE:
        start, end = req, req + 1
    else:
        start, end = tl.load(Cu + req), tl.load(Cu + req + 1)
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    j = tl.arange(0, BW)
    pos = end - start - WIDTH + j
    mask = (c[:, None] < C) & (j[None, :] < WIDTH)
    old = tl.load(S + (slot * C + c[:, None]) * WIDTH + (pos[None, :] + WIDTH),
                  mask & (pos[None, :] < 0), other=0)
    new = tl.load(X + (start + pos[None, :]) * C + c[:, None],
                  mask & (pos[None, :] >= 0), other=0)
    # Each program owns complete histories for its channels: no in-place
    # shift can race with a different program's read.
    tl.store(S + (slot * C + c[:, None]) * WIDTH + j[None, :],
             tl.where(pos[None, :] >= 0, new, old), mask)


def packed_causal_conv(x, weight, states, slots, cu_seqlens, max_len, *, decode=False):
    """Read persistent left context; update raw-QKV histories after convolution.

    Request slots must be unique, as guaranteed by the scheduler. Two launches
    keep history updates from racing with convolution reads across token tiles.
    """
    channels, _, kernel = weight.shape
    output = torch.empty_like(x)
    _conv[(triton.cdiv(max_len, 4), triton.cdiv(channels, 128), slots.numel())](
        x, weight, states, slots, cu_seqlens, output, channels, kernel,
        weight.stride(0), weight.stride(2), decode, 4, 128)
    if kernel > 1:
        _conv_state[(triton.cdiv(channels, 64), slots.numel())](
            x, states, slots, cu_seqlens, channels, kernel - 1, decode,
            64, triton.next_power_of_2(kernel - 1))
    return output


@triton.jit
def _decode(Q, Kptr, Vptr, A, Bptr, Log, Bias, Pool, Slots, Out,
            HQ: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    rows = tl.program_id(2) * BV + tl.arange(0, BV)
    cols = tl.arange(0, BK)
    slot = tl.load(Slots + batch)
    qhead = head // (HV // HQ)
    q = tl.load(Q + (batch * HQ + qhead) * DK + cols, cols < DK, other=0).to(tl.float32)
    k = tl.load(Kptr + (batch * HQ + qhead) * DK + cols, cols < DK, other=0).to(tl.float32)
    q = q * tl.rsqrt(tl.sum(q * q, 0) + 1e-6) * (DK ** -0.5)
    k = k * tl.rsqrt(tl.sum(k * k, 0) + 1e-6)
    a = tl.load(A + batch * HV + head).to(tl.float32)
    b = tl.load(Bptr + batch * HV + head).to(tl.float32)
    x = a + tl.load(Bias + head).to(tl.float32)
    softplus = tl.where(x > 20, x, tl.log(1 + tl.exp(tl.minimum(x, 20))))
    decay = tl.exp(-tl.exp(tl.load(Log + head).to(tl.float32)) * softplus)
    beta = tl.sigmoid(b)
    ptr = Pool + ((slot * HV + head) * DV + rows[:, None]) * DK + cols[None, :]
    mask = (rows[:, None] < DV) & (cols[None, :] < DK)
    h = tl.load(ptr, mask, other=0) * decay
    v = tl.load(Vptr + (batch * HV + head) * DV + rows, rows < DV, other=0).to(tl.float32)
    delta = (v - tl.sum(h * k[None, :], 1)) * beta
    h = h + delta[:, None] * k[None, :]
    tl.store(ptr, h, mask)
    y = tl.sum(h * q[None, :], 1)
    tl.store(Out + (batch * HV + head) * DV + rows, y, rows < DV)


def indexed_gdn_decode(q, k, v, a, b, a_log, bias, pool, slots):
    batch, _, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    output = torch.empty_like(v)
    _decode[(batch, value_heads, triton.cdiv(value_dim, 8))](
        q, k, v, a, b, a_log, bias, pool, slots, output,
        heads, value_heads, key_dim, value_dim, triton.next_power_of_2(key_dim), 8,
        enable_fp_fusion=False)
    return output
