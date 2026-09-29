"""Gated DeltaNet linear attention layer for Qwen3.5 (text-only).

Kernels follow SGLang's GDN design, with no pure-torch fallback:
    * prefill (extend): the vendored FLA Triton chunk kernel, with packed
      variable lengths and persistent state-pool indexing. Final recurrent
      states are written into the pool in place.
    * decode: indexed Triton FP32 state-pool kernel for K=V=128;
      FlashInfer ``gated_delta_rule_decode_pretranspose`` remains the fallback
      (``flashinfer.gdn_decode``), which computes the sigmoid gating
      (g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)) inside
      the kernel from the raw ``a`` / ``b`` projections.

Requires CUDA; q/k/v must be bf16/fp16. ModelRunner warms up the kernels
before CUDA Graph capture.

State layout: V-major / K-last ``[N, HV, V, K]``  the SGLang and FlashInfer
convention (K-last).

Layer forward math:
    mixed = silu(conv1d(in_proj_qkv(x)))            # causal depthwise conv
    q, k, v = split(mixed)                          # -> heads
    a, b, z  = in_proj_a/b/z(x)
    y      = delta_rule(q, k, v, a, b, A_log, dt_bias, state)
    y      = RMSNormGated(y, z)
    out    = out_proj(y)

State (conv_states / recurrent_states) is allocated by GatedDeltaNet under
the ModelRunner-managed request lifecycle. Each sequence reads/writes its own
persistent slot via context.state_indices.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from hybridinfer.layers.layernorm import RMSNormGated
from hybridinfer.layers.gdn_kernels import (packed_causal_conv, indexed_gdn_decode,
                                           compact_conv_endpoints, packed_gdn_recurrent)
from hybridinfer.utils.context import get_context


# ---------------------------------------------------------------------------
# Delta-rule kernels (no torch fallback)
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chunked delta rule prefill via the SGLang Triton chunk kernel.

    Args:
        query/key/value: (B, S, H, Dk / Dv) bf16/fp16, head-last.
        g: (B, S, H) per-step log decay (fp32 preferred; cast internally)
        beta: (B, S, H) write gate
        initial_state: (N, H, V, K) K-last state pool
        initial_state_indices: (B,) request slots into the state pool
        cu_seqlens: (B + 1,) packed sequence boundaries for variable lengths
    Returns:
        out: (B, S, H, Dv).  The final recurrent state is written in-place
        into ``initial_state`` by the kernel (INPLACE_UPDATE epilogue); the
        kernel's per-chunk ``h`` tensor only holds states *entering* each
        chunk, so it must not be used as the final state.
    """
    from hybridinfer.layers.fla.chunk import chunk_gated_delta_rule as fla_chunk

    assert query.dtype != torch.float32, "Triton chunk kernel requires bf16/fp16 q/k/v"
    # The FLA kernels hard-code contiguous strides (e.g. stride_v = H*V); a
    # strided view (e.g. v split from the packed conv output) would be read
    # at wrong addresses. Materialize contiguous copies — q/k usually already
    # are (GQA repeat_interleave), v is the one that needs the copy.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    o, _, _ = fla_chunk(
        q=query,
        k=key,
        v=value,
        g=g.to(torch.float32),
        beta=beta.to(torch.float32),
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_qk_l2norm_in_kernel=True,
    )
    return o


def decode_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token delta rule decode via FlashInfer (in-kernel gating).

    Args:
        query/key/value: (B, S, H, Dk / Dv) bf16/fp16, S must be 1.
        a: (B, S, Hv) raw input-dependent decay projection
        b: (B, S, Hv) raw update-gate projection
        A_log: (Hv,) log-space decay parameter (fp32)
        dt_bias: (Hv,) time-step bias
        initial_state: (B, H, V, K) K-last (fp32 legacy path, or bf16 for
            K=V=128 and T<=4)
    Returns:
        out: (B, S, H, Dv), final_state: (B, H, V, K) K-last
    """
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    B, S, H, K = query.shape
    _, _, HV, V = value.shape
    assert S == 1, f"FlashInfer decode requires S=1, got S={S}"
    out, new_state = gated_delta_rule_decode_pretranspose(
        q=query.view(B, 1, H, K),
        k=key.view(B, 1, H, K),
        v=value.view(B, 1, HV, V),
        state=initial_state.contiguous(),
        A_log=A_log.detach().to(torch.float32),
        a=a.view(B, 1, HV),
        dt_bias=dt_bias.detach(),
        b=b.view(B, 1, HV),
        scale=None,
        output=None,
        use_qk_l2norm=True,
    )
    return out.view(B, S, HV, V), new_state


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------


class GatedDeltaNet(nn.Module):

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.layer_idx = layer_idx
        self.gqa_ratio = self.num_v_heads // self.num_k_heads
        state_dtype = getattr(config, "mamba_ssm_dtype", "float32")
        supported_state_dtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                                  torch.float32: torch.float32, torch.bfloat16: torch.bfloat16}
        if state_dtype not in supported_state_dtypes:
            raise ValueError(f"Unsupported GDN recurrent state dtype: {state_dtype!r}")
        self.recurrent_state_dtype = supported_state_dtypes[state_dtype]
        self.decode_backend = os.environ.get("HYBRIDINFER_GDN_DECODE_BACKEND", "pool")
        if self.decode_backend not in ("pool", "flashinfer"):
            raise ValueError("HYBRIDINFER_GDN_DECODE_BACKEND must be pool or flashinfer")

        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.norm.weight = nn.Parameter(self.norm.weight.data.to(torch.float32))
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        # State pools, allocated by ModelRunner after init.
        self.conv_states: torch.Tensor = torch.tensor([])
        self.recurrent_states: torch.Tensor = torch.tensor([])

    def allocate_state_pool(self, num_slots: int):
        """Allocate this layer's persistent runtime state on its CUDA device."""
        device = self.in_proj_qkv.weight.device
        if device.type != "cuda":
            raise RuntimeError("GatedDeltaNet state pool must be allocated on CUDA")

        # The convolution cache stores raw qkv projection values, so it uses
        # the projection/compute dtype. Recurrent states obey the checkpoint
        # configuration; FP32 prevents an extra BF16 truncation at scheduler
        # chunk boundaries that does not occur inside a one-shot chunk scan.
        # FlashInfer supports BF16 state for K=V=128; other head layouts use
        # its legacy FP32 state path. The existing debug override stays valid.
        conv_dtype = self.in_proj_qkv.weight.dtype
        fp32_state_debug = os.environ.get(
            "HYBRIDINFER_GDN_FP32_STATE", ""
        ).lower() in ("1", "true", "yes")
        recurrent_dtype = (
            torch.float32
            if fp32_state_debug
            or self.head_k_dim != 128
            or self.head_v_dim != 128
            else self.recurrent_state_dtype
        )
        self.conv_states = torch.zeros(
            num_slots,
            self.conv_dim,
            self.conv_kernel_size - 1,
            dtype=conv_dtype,
            device=device,
        )
        self.recurrent_states = torch.zeros(
            num_slots,
            self.num_v_heads,
            self.head_v_dim,
            self.head_k_dim,
            dtype=recurrent_dtype,
            device=device,
        )

    def reset_state(self, slots: int | list[int] | tuple[int, ...] | torch.Tensor):
        """Clear one or more persistent request slots entirely on the GPU."""
        if self.conv_states.numel() == 0 or self.recurrent_states.numel() == 0:
            raise RuntimeError("GatedDeltaNet state pool has not been allocated")

        if isinstance(slots, torch.Tensor):
            slot_indices = slots.to(
                device=self.conv_states.device,
                dtype=torch.int64,
            )
        else:
            slot_indices = torch.as_tensor(
                slots,
                device=self.conv_states.device,
                dtype=torch.int64,
            )
        slot_indices = slot_indices.reshape(-1)
        if slot_indices.numel() == 0:
            return
        self.conv_states.index_fill_(0, slot_indices, 0)
        self.recurrent_states.index_fill_(0, slot_indices, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill:
            return self._forward_prefill(hidden_states)
        else:
            return self._forward_decode(hidden_states)

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3 or hidden_states.shape[0] != 1:
            raise ValueError("GDN prefill expects [1, total_tokens, hidden_size]")
        return self.forward_core_from_dense(self.forward_dense_pre(hidden_states.squeeze(0)))

    def forward_dense_pre(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        return raw_qkv, z, b, a

    def forward_core_from_dense(
        self,
        attention_pre: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        raw_qkv_packed, z, b, a = attention_pre
        context = get_context()
        if not context.is_prefill:
            return self._forward_decode_dense(attention_pre)
        if context.prefill_slices is None or context.cu_seqlens_q is None:
            raise RuntimeError(
                "GDN prefill requires packed slices and cu_seqlens"
            )

        total_tokens = raw_qkv_packed.shape[0]
        if context.state_indices is None or self.recurrent_states.numel() == 0:
            raise RuntimeError("GDN prefill requires allocated state pools and state_indices")
        descriptor = context.batch_descriptor
        if descriptor is not None and descriptor.mode == "spec_decode":
            return self._forward_verify(attention_pre)
        qkv = packed_causal_conv(
            raw_qkv_packed, self.conv1d.weight, self.conv_states,
            context.state_indices, context.cu_seqlens_q,
            context.max_seqlen_q or max(end - start for start, end in context.prefill_slices),
        ).unsqueeze(0)

        query = qkv[..., : self.key_dim].reshape(1, total_tokens, -1, self.head_k_dim)
        key = qkv[..., self.key_dim:self.key_dim * 2].reshape(
            1, total_tokens, -1, self.head_k_dim
        )
        value = qkv[..., self.key_dim * 2:].reshape(
            1, total_tokens, -1, self.head_v_dim
        )
        z = z.unsqueeze(0).reshape(1, total_tokens, -1, self.head_v_dim)
        b = b.unsqueeze(0).reshape(1, total_tokens, -1)
        a = a.unsqueeze(0).reshape(1, total_tokens, -1)
        beta = torch.sigmoid(b)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        if self.gqa_ratio > 1:
            query = query.repeat_interleave(self.gqa_ratio, dim=2)
            key = key.repeat_interleave(self.gqa_ratio, dim=2)

        out = chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=self.recurrent_states,
            initial_state_indices=context.state_indices,
            cu_seqlens=context.cu_seqlens_q,
            chunk_indices=context.prefill_chunk_indices,
        )
        out = out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        out = self.norm(out, z)
        out = out.reshape(1, total_tokens, -1)

        return self.out_proj(out)

    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._forward_decode_dense(self.forward_dense_pre(hidden_states))

    def _forward_decode_dense(self, attention_pre) -> torch.Tensor:
        # Same stateful decode kernel for native and partitioned execution.
        raw_qkv, z, b, a = attention_pre
        B = raw_qkv.shape[0]
        raw_qkv = raw_qkv.reshape(B, -1)
        b, a = b.reshape(B, 1, -1), a.reshape(B, 1, -1)
        idx = get_context().state_indices
        if idx is None or idx.numel() == 0:
            raise RuntimeError(
                "GDN decode requires Context.state_indices; "
                "ModelRunner must pass persistent request slots"
            )
        use_pool = (self.recurrent_states.dtype == torch.float32
                    and self.head_k_dim == 128 and self.head_v_dim == 128
                    and self.decode_backend == "pool")
        out = packed_causal_conv(raw_qkv, self.conv1d.weight, self.conv_states,
                                 idx, None, 1, decode=True,
                                 round_before_silu=False).unsqueeze(1)

        query, key, value = torch.split(
            out, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(B, 1, -1, self.head_k_dim)  # (B, S, Hk, Dk)
        key = key.reshape(B, 1, -1, self.head_k_dim)
        value = value.reshape(B, 1, -1, self.head_v_dim)

        z = z.reshape(B, 1, -1, self.head_v_dim)
        if self.gqa_ratio > 1 and not use_pool:
            query = query.repeat_interleave(self.gqa_ratio, dim=2)
            key = key.repeat_interleave(self.gqa_ratio, dim=2)

        if use_pool:
            out = indexed_gdn_decode(query.contiguous(), key.contiguous(), value.contiguous(),
                                     a, b, self.A_log, self.dt_bias, self.recurrent_states, idx)
        else:
            rec_state = self.recurrent_states.index_select(0, idx)
            out, new_rec = decode_gated_delta_rule(
                query, key, value, a, b, self.A_log, self.dt_bias, rec_state,
            )
            self.recurrent_states.index_copy_(0, idx, new_rec)

        # Per-head gated norm, then merge heads -> value_dim (matches official:
        # core_attn_out.reshape(-1, head_v_dim) -> norm -> reshape(batch, seq, -1))
        out = out.reshape(-1, self.head_v_dim)  # (B*Hv, Dv)
        z = z.reshape(-1, self.head_v_dim)  # (B*Hv, Dv)
        out = self.norm(out, z)
        out = out.reshape(B, 1, -1)  # (B, S, value_dim)
        return self.out_proj(out)

    def _forward_verify(self, attention_pre):
        """Short recurrent update, retaining the original trial's endpoints.

        Dense projections are packed once; recurrence uses the same one-token
        arithmetic as decode, retaining per-token states inside a fused scan.
        """
        raw, z, b, a = attention_pre
        ctx = get_context()
        if self.recurrent_states.dtype == torch.float32 and self.decode_backend == "pool":
            conv = compact_conv_endpoints(raw, self.conv_states, ctx.state_indices,
                ctx.cu_seqlens_q, ctx.batch_descriptor.max_query_len)
            mixed = packed_causal_conv(raw, self.conv1d.weight, self.conv_states,
                ctx.state_indices, ctx.cu_seqlens_q, ctx.batch_descriptor.max_query_len,
                round_before_silu=False)
            q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], -1)
            q = q.reshape(raw.shape[0], self.num_k_heads, self.head_k_dim).contiguous()
            k = k.reshape_as(q).contiguous()
            v = v.reshape(raw.shape[0], self.num_v_heads, self.head_v_dim).contiguous()
            outputs, recurrent = packed_gdn_recurrent(q, k, v, a, b, self.A_log, self.dt_bias,
                self.recurrent_states, ctx.state_indices, ctx.cu_seqlens_q)
            if ctx.state_endpoints is not None:
                ctx.state_endpoints[self.layer_idx] = (self, conv, recurrent)
            out = self.norm(outputs.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
            return self.out_proj(out.reshape(1, raw.shape[0], self.value_dim))
        outputs = torch.empty((raw.shape[0], self.num_v_heads, self.head_v_dim),
                              dtype=raw.dtype, device=raw.device)
        conv = torch.empty((raw.shape[0], *self.conv_states.shape[1:]),
                           dtype=self.conv_states.dtype, device=raw.device)
        recurrent = torch.empty((raw.shape[0], *self.recurrent_states.shape[1:]),
                                dtype=self.recurrent_states.dtype, device=raw.device)
        for row, (start, end) in enumerate(ctx.prefill_slices):
            slot = ctx.state_indices[row:row+1]
            for t in range(start, end):
                qkv = packed_causal_conv(raw[t:t+1], self.conv1d.weight,
                                        self.conv_states, slot, None, 1, decode=True, round_before_silu=False)
                q, k, v = torch.split(qkv, [self.key_dim, self.key_dim, self.value_dim], -1)
                q = q.reshape(1, 1, self.num_k_heads, self.head_k_dim).contiguous()
                k = k.reshape_as(q).contiguous()
                v = v.reshape(1, 1, self.num_v_heads, self.head_v_dim).contiguous()
                if (self.recurrent_states.dtype == torch.float32
                        and self.head_k_dim == self.head_v_dim == 128
                        and self.decode_backend == "pool"):
                    out = indexed_gdn_decode(q, k, v, a[t:t+1], b[t:t+1],
                        self.A_log, self.dt_bias, self.recurrent_states, slot)
                else:
                    q = q.repeat_interleave(self.gqa_ratio, 2)
                    k = k.repeat_interleave(self.gqa_ratio, 2)
                    out, state = decode_gated_delta_rule(q, k, v, a[t:t+1].clone(), b[t:t+1].clone(),
                        self.A_log, self.dt_bias, self.recurrent_states.index_select(0, slot))
                    self.recurrent_states.index_copy_(0, slot, state)
                outputs[t:t+1].copy_(out.reshape(1, self.num_v_heads, self.head_v_dim))
                conv[t:t+1].copy_(self.conv_states.index_select(0, slot))
                recurrent[t:t+1].copy_(self.recurrent_states.index_select(0, slot))
        if ctx.state_endpoints is not None:
            ctx.state_endpoints[self.layer_idx] = (self, conv, recurrent)
        out = self.norm(outputs.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(out.reshape(1, raw.shape[0], self.value_dim))
