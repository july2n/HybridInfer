"""Packed causal convolution and FP32 indexed GDN decode kernels."""
from functools import lru_cache

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(X, W, S, Slots, Cu, Y, C: tl.constexpr, K: tl.constexpr,
          WS: tl.constexpr, WK: tl.constexpr, DECODE: tl.constexpr,
          BT: tl.constexpr, BC: tl.constexpr, ROUND_BEFORE_SILU: tl.constexpr):
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
        value = tl.where(pos[:, None] >= 0, current, history)
        weight = tl.load(W + c * WS + i * WK, c < C, other=0)
        # vLLM retains the projection dtype for the product before accumulation.
        if ROUND_BEFORE_SILU:
            value = value.to(tl.float32)
            weight = weight.to(tl.float32)
        acc += value * weight[None, :]
    # Preserve ordinary HF-compatible rounding; speculative uses vLLM arithmetic.
    if ROUND_BEFORE_SILU:
        rounded = acc.to(X.dtype.element_ty).to(tl.float32)
        result = rounded * tl.sigmoid(rounded)
    else:
        # vLLM speculative convolution applies SiLU to the FP32 accumulator.
        result = acc / (1 + tl.exp(-acc))
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


def packed_causal_conv(x, weight, states, slots, cu_seqlens, max_len, *, decode=False, round_before_silu=True):
    """Read persistent left context; update raw-QKV histories after convolution.

    Request slots must be unique, as guaranteed by the scheduler. Two launches
    keep history updates from racing with convolution reads across token tiles.
    """
    channels, _, kernel = weight.shape
    output = torch.empty_like(x)
    _conv[(triton.cdiv(max_len, 4), triton.cdiv(channels, 128), slots.numel())](
        x, weight, states, slots, cu_seqlens, output, channels, kernel,
        weight.stride(0), weight.stride(2), decode, 4, 128, round_before_silu)
    if kernel > 1:
        _conv_state[(triton.cdiv(channels, 64), slots.numel())](
            x, states, slots, cu_seqlens, channels, kernel - 1, decode,
            64, triton.next_power_of_2(kernel - 1))
    return output


@lru_cache(maxsize=32)
def _decode_offsets(batch, device):
    return torch.arange(batch+1, dtype=torch.int32, device=device)


def indexed_gdn_decode(q, k, v, a, b, a_log, bias, pool, slots):
    batch, _, heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    output = torch.empty_like(v)
    # Deliberately retain the packed endpoint store: removing it changes
    # compiler fusion and FP32 state bits, even when BF16 outputs agree.
    # This transient scratch costs batch*HV*DV*DK*sizeof(state) bytes.
    snapshots = torch.empty((batch, value_heads, value_dim, key_dim),
                            dtype=pool.dtype, device=pool.device)
    # Reuse the recurrent scan itself: matching BV/FMA alone still lets two
    # separately compiled bodies fuse expressions differently.
    _packed_recurrent[(batch, value_heads, triton.cdiv(value_dim, 32))](
        q, k, v, a, b, a_log, bias, pool, slots,
        _decode_offsets(batch, q.device), output, snapshots,
        heads, value_heads, key_dim, value_dim, triton.next_power_of_2(key_dim), 32,
        enable_fp_fusion=True, num_stages=3, num_warps=4)
    return output


@triton.jit
def _conv_endpoints(X, S, Slots, Cu, Snapshots, C: tl.constexpr,
                    WIDTH: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    req = tl.program_id(1)
    slot = tl.load(Slots+req)
    start, end = tl.load(Cu+req), tl.load(Cu+req+1)
    c = tl.program_id(0)*BC+tl.arange(0, BC)
    j = tl.arange(0, BW)
    mask = (c[:, None] < C) & (j[None, :] < WIDTH)
    for t in range(start, end):
        pos = t-start+1-WIDTH+j
        old = tl.load(S+(slot*C+c[:, None])*WIDTH+(pos[None, :]+WIDTH),
                      mask & (pos[None, :] < 0), other=0)
        new = tl.load(X+(start+pos[None, :])*C+c[:, None],
                      mask & (pos[None, :] >= 0), other=0)
        tl.store(Snapshots+(t*C+c[:, None])*WIDTH+j[None, :],
                 tl.where(pos[None, :] >= 0, new, old), mask)


def conv_endpoints(raw, pool, slots, cu_seqlens):
    snapshots = torch.empty((raw.shape[0], *pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    channels, width = pool.shape[1:]
    _conv_endpoints[(triton.cdiv(channels, 64), slots.numel())](
        raw, pool, slots, cu_seqlens, snapshots, channels, width, 64,
        triton.next_power_of_2(width))
    return snapshots


@triton.jit
def _conv_history(X, Pool, Slots, Cu, History, C: tl.constexpr,
                  WIDTH: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr):
    req = tl.program_id(2)
    start, end = tl.load(Cu+req), tl.load(Cu+req+1)
    slot = tl.load(Slots+req)
    t = tl.program_id(0)*BT+tl.arange(0, BT)
    c = tl.program_id(1)*BC+tl.arange(0, BC)
    mask = (t[:, None] < end-start+WIDTH) & (c[None, :] < C)
    old = tl.load(Pool+(slot*C+c[None, :])*WIDTH+t[:, None],
                  mask & (t[:, None] < WIDTH), other=0)
    new = tl.load(X+(start+t[:, None]-WIDTH)*C+c[None, :],
                  mask & (t[:, None] >= WIDTH), other=0)
    tl.store(History+(start+req*WIDTH+t[:, None])*C+c[None, :],
             tl.where(t[:, None] < WIDTH, old, new), mask)


@triton.jit
def _select_conv_history(History, Cu, Indices, Slots, Pool,
                         N: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr,
                         BN: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    row = tl.program_id(1)
    endpoint = tl.load(Indices+row)
    requests = tl.arange(0, BN)
    ends = tl.load(Cu+requests+1, requests < N, other=2147483647)
    req = tl.sum((endpoint >= ends).to(tl.int32), 0)
    slot = tl.load(Slots+row)
    c = tl.program_id(0)*BC+tl.arange(0, BC)
    j = tl.arange(0, BW)
    mask = (c[:, None] < C) & (j[None, :] < WIDTH)
    value = tl.load(History+(endpoint+req*WIDTH+1+j[None, :])*C+c[:, None], mask, other=0)
    tl.store(Pool+(slot*C+c[:, None])*WIDTH+j[None, :], value, mask)


class ConvHistory:
    """One initial history plus raw trial tokens per request; windows are views logically."""
    def __init__(self, history, cu, width):
        self.history, self.cu, self.width = history, cu, width

    def commit(self, pool, slots, indices):
        channels = self.history.shape[1]
        requests = self.cu.numel()-1
        _select_conv_history[(triton.cdiv(channels, 64), indices.numel())](
            self.history, self.cu, indices, slots, pool, requests, channels,
            self.width, triton.next_power_of_2(requests), 64,
            triton.next_power_of_2(self.width))

    def index_select(self, dim, indices):
        if dim != 0:
            raise ValueError('ConvHistory only supports endpoint selection')
        result = torch.empty((indices.numel(), self.history.shape[1], self.width),
                             dtype=self.history.dtype, device=self.history.device)
        self.commit(result, torch.arange(indices.numel(), device=indices.device), indices)
        return result

    def __getitem__(self, index):
        indices = torch.tensor([index], device=self.history.device)
        return self.index_select(0, indices)[0]


def compact_conv_endpoints(raw, pool, slots, cu_seqlens, max_len):
    channels, width = pool.shape[1:]
    history = torch.empty((raw.shape[0]+slots.numel()*width, channels),
                          dtype=pool.dtype, device=pool.device)
    _conv_history[(triton.cdiv(max_len+width, 4), triton.cdiv(channels, 128), slots.numel())](
        raw, pool, slots, cu_seqlens, history, channels, width, 128, 4)
    return ConvHistory(history, cu_seqlens, width)


@triton.jit
def _packed_recurrent(Q, Kptr, Vptr, A, Bptr, Log, Bias, Pool, Slots, Cu, Out, Snapshots,
                      HQ: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
                      BK: tl.constexpr, BV: tl.constexpr, SAVE_STATES: tl.constexpr = True):
    batch, head = tl.program_id(0), tl.program_id(1)
    rows = tl.program_id(2)*BV+tl.arange(0, BV)
    cols = tl.arange(0, BK)
    slot = tl.load(Slots+batch)
    start, end = tl.load(Cu+batch), tl.load(Cu+batch+1)
    qhead = head//(HV//HQ)
    ptr = Pool+((slot*HV+head)*DV+rows[:, None])*DK+cols[None, :]
    mask = (rows[:, None] < DV) & (cols[None, :] < DK)
    h = tl.load(ptr, mask, other=0)
    for t in range(start, end):
        q = tl.load(Q+(t*HQ+qhead)*DK+cols, cols < DK, other=0).to(tl.float32)
        k = tl.load(Kptr+(t*HQ+qhead)*DK+cols, cols < DK, other=0).to(tl.float32)
        q = q*tl.rsqrt(tl.sum(q*q, 0)+1e-6)*(DK**-0.5)
        k = k*tl.rsqrt(tl.sum(k*k, 0)+1e-6)
        a = tl.load(A+t*HV+head).to(tl.float32)
        b = tl.load(Bptr+t*HV+head).to(tl.float32)
        x = a+tl.load(Bias+head).to(tl.float32)
        softplus = tl.where(x > 20, x, tl.log(1+tl.exp(tl.minimum(x, 20))))
        decay = tl.exp(-tl.exp(tl.load(Log+head).to(tl.float32))*softplus)
        beta = tl.sigmoid(b)
        h = h*decay
        v = tl.load(Vptr+(t*HV+head)*DV+rows, rows < DV, other=0).to(tl.float32)
        delta = (v-tl.sum(h*k[None, :], 1))*beta
        h = h+delta[:, None]*k[None, :]
        if SAVE_STATES:
            tl.store(Snapshots+((t*HV+head)*DV+rows[:, None])*DK+cols[None, :], h, mask)
        y = tl.sum(h*q[None, :], 1)
        tl.store(Out+(t*HV+head)*DV+rows, y, rows < DV)
    tl.store(ptr, h, mask)


def packed_gdn_recurrent(q, k, v, a, b, a_log, bias, pool, slots, cu_seqlens):
    if pool.dtype != torch.float32:
        raise ValueError('Packed recurrent kernel requires FP32 state')
    total, hq, dk = q.shape
    hv, dv = v.shape[1:]
    output = torch.empty_like(v)
    snapshots = torch.empty((total, hv, dv, dk), dtype=pool.dtype, device=pool.device)
    _packed_recurrent[(slots.numel(), hv, triton.cdiv(dv, 32))](
        q, k, v, a, b, a_log, bias, pool, slots, cu_seqlens, output, snapshots,
        hq, hv, dk, dv, triton.next_power_of_2(dk), 32, enable_fp_fusion=True, num_stages=3, num_warps=4)
    return output, snapshots
