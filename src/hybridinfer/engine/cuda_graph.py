from __future__ import annotations

from typing import TYPE_CHECKING
from enum import Enum
from contextlib import contextmanager

from .compilation import SplitGraph

import torch

from hybridinfer.utils.context import get_context, reset_context, set_context


if TYPE_CHECKING:
    from .model_runner import ModelRunner


class CUDAGraphMode(str, Enum):
    NONE = "none"
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"


@contextmanager
def graph_mode(mode):
    context = get_context()
    previous = context.cudagraph_mode
    context.cudagraph_mode = mode.value
    try:
        yield
    finally:
        context.cudagraph_mode = previous


class CUDAGraphWrapper:
    """Capture/replay in one runtime mode, pass through in all other modes.

    Capture is explicit after warmup. The caller owns static input preparation;
    outputs and graph records are owned by the wrapper. Mismatched modes never
    capture or replay, so FULL capture can safely execute PIECEWISE wrappers.
    """

    def __init__(self, runnable, mode, prepare=None):
        if mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            raise ValueError("wrapper mode must be FULL or PIECEWISE")
        self.runnable = runnable
        self.mode = mode
        self.prepare = prepare
        self.graphs = {}
        self.outputs = {}

    def __call__(self, *args, graph_key=None):
        if get_context().cudagraph_mode != self.mode.value or graph_key not in self.graphs:
            return self.runnable(*args)
        if self.prepare is not None:
            self.prepare(graph_key, *args)
        self.graphs[graph_key].replay()
        return self.outputs[graph_key]

    def capture(self, key, args, pool=None, transform=None):
        if get_context().cudagraph_mode != self.mode.value:
            raise RuntimeError("capture runtime mode does not match wrapper")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool):
            output = self.runnable(*args)
            if transform is not None:
                output = transform(output)
        self.graphs[key] = graph
        self.outputs[key] = output
        return graph, output

    def clear(self):
        self.graphs.clear()
        self.outputs.clear()


class CUDAGraphDispatcher:
    """Default FULL_AND_PIECEWISE policy with explicit supported buckets."""

    def __init__(self, manager):
        self.manager = manager

    def dispatch(self, descriptor, is_prefill, num_tokens):
        manager = self.manager
        if getattr(manager.runner, "enforce_eager", False):
            return CUDAGraphMode.NONE
        # Current full graph kernels support single-token decode only. Spec
        # verification stays outside this dispatcher until its state is captured.
        if descriptor is not None and descriptor.mode in ("spec_decode", "draft"):
            return CUDAGraphMode.NONE
        uniform = (not is_prefill and (descriptor is None or (
            descriptor.mode == "decode" and descriptor.uniform_token_count == 1
            and descriptor.max_query_len == 1 and descriptor.num_tokens == descriptor.num_reqs)))
        if uniform and manager.policy in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE) and num_tokens in manager.decode_graphs:
            return CUDAGraphMode.FULL
        if (is_prefill or uniform) and manager.policy in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE) and manager.runner.use_prefill_cudagraph and any(
            size >= num_tokens and size in manager.prefill_graphs for size in manager.prefill_graph_sizes
        ):
            return CUDAGraphMode.PIECEWISE
        return CUDAGraphMode.NONE


class CudaGraphManager:
    """Own decode and piecewise-prefill CUDA graph capture/replay."""

    def __init__(self, runner: "ModelRunner") -> None:
        self.runner = runner
        self.split_graph = SplitGraph(self)
        self.full_wrapper = CUDAGraphWrapper(
            self.split_graph, CUDAGraphMode.FULL, prepare=self._prepare_full_inputs)
        self.dispatcher = CUDAGraphDispatcher(self)

        self.decode_graph_sizes: list[int] = []
        self.decode_graphs = self.full_wrapper.graphs
        self.decode_graph_pool = None
        self.decode_graph_vars: dict[str, torch.Tensor] | None = None

        self.prefill_graph_sizes: list[int] = []
        self.prefill_graphs: dict[int, dict[int, dict]] = {}
        self.prefill_graph_pool = None
        self.piecewise_buffers: dict | None = None
        self._piecewise_callables = self.split_graph.backend.callables

    @property
    def model(self):
        return self.runner.model

    @property
    def config(self):
        return self.runner.config

    @property
    def policy(self):
        config = getattr(self.runner, "config", None)
        return CUDAGraphMode(getattr(config, "cudagraph_mode", "FULL_AND_PIECEWISE").lower())

    @torch.inference_mode()
    def capture(self) -> None:
        if self.policy in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE):
            self._capture_decode()
        if self.runner.use_prefill_cudagraph and self.policy in (
            CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE
        ):
            self.capture_prefill()

    def _capture_decode(self) -> None:
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.runner.block_size - 1
        ) // self.runner.block_size
        device = torch.cuda.current_device()
        input_ids = torch.zeros(max_bs, dtype=torch.int64, device=device)
        positions = torch.zeros(max_bs, dtype=torch.int64, device=device)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32, device=device)
        context_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        block_tables = torch.zeros(
            max_bs,
            max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        state_indices = torch.arange(max_bs, dtype=torch.int64, device=device)
        outputs = torch.zeros(
            max_bs,
            hf_config.hidden_size,
            device=device,
        )

        # Cover common small batches exactly; padding would need isolated
        # GDN scratch slots because decode updates recurrent state in place.
        self.decode_graph_sizes = sorted({
            size for size in list(range(1, min(max_bs, 16) + 1))
            + [max_bs] + list(range(32, max_bs + 1, 16))
            if 0 < size <= max_bs
        })
        self.full_wrapper.clear()
        self.decode_graphs = self.full_wrapper.graphs
        self.decode_graph_pool = None

        for bs in reversed(self.decode_graph_sizes):
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                state_indices=state_indices[:bs],
            )
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    outputs[:bs] = self.full_wrapper(input_ids[:bs], positions[:bs])
            torch.cuda.current_stream().wait_stream(warmup_stream)

            # FULL wrapper is active; every inner PIECEWISE wrapper passes through.
            with graph_mode(CUDAGraphMode.FULL):
                graph, _ = self.full_wrapper.capture(
                    bs, (input_ids[:bs], positions[:bs]), self.decode_graph_pool,
                    transform=lambda hidden, bs=bs: self._copy_full_output(outputs, hidden, bs),
                )
            if self.decode_graph_pool is None:
                self.decode_graph_pool = graph.pool()
            self.decode_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.decode_graph_vars = {
            "input_ids": input_ids,
            "positions": positions,
            "slot_mapping": slot_mapping,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "state_indices": state_indices,
            "outputs": outputs,
        }

    def capture_prefill(
        self,
        buckets: list[int] | None = None,
    ) -> None:
        max_prefill_tokens = self.config.max_num_batched_tokens
        if buckets is None:
            buckets = [
                size
                for size in (128, 256, 512, 768, 1024, 1536, 2048)
                if size <= max_prefill_tokens
            ]
            if max_prefill_tokens > 0 and max_prefill_tokens not in buckets:
                buckets.append(max_prefill_tokens)

        if any(size <= 0 for size in buckets):
            raise ValueError("prefill graph buckets must be positive")
        buckets = sorted(set(buckets))
        self.prefill_graph_sizes = buckets
        self.clear_prefill()
        if not self.runner.use_prefill_cudagraph or not buckets:
            return

        self.prefill_graph_pool = torch.cuda.graph_pool_handle()
        self.piecewise_buffers = self._allocate_piecewise_buffers(
            max(buckets),
            self.config.hf_config.hidden_size,
        )

        for graph_size in buckets:
            bucket_entries = {}
            for layer_idx, layer in enumerate(self.model.model.layers):
                bucket_entries[layer_idx] = self.capture_prefill_layer_segments(
                    layer,
                    graph_size,
                    has_residual=layer_idx > 0,
                )
            self.prefill_graphs[graph_size] = bucket_entries
            torch.cuda.synchronize()

    @staticmethod
    def _copy_full_output(outputs, hidden, bs):
        outputs[:bs].copy_(hidden)
        return outputs[:bs]

    def run(self, input_ids, positions, is_prefill):
        mode = self.dispatcher.dispatch(get_context().batch_descriptor, is_prefill, input_ids.size(0))
        with graph_mode(mode):
            hidden = self.full_wrapper(input_ids, positions, graph_key=input_ids.size(0))
        return self.runner.compute_logits(hidden, is_prefill)

    def run_prefill(self, input_ids, positions):
        with graph_mode(CUDAGraphMode.PIECEWISE):
            hidden = self.full_wrapper(input_ids, positions)
        return self.runner.compute_logits(hidden, True)

    def _run_piecewise_hidden(self, input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        if context.is_prefill and context.prefill_slices is None:
            raise RuntimeError("prefill graph requires Context.prefill_slices")

        num_tokens = input_ids.size(0)
        graph_size = next(
            (size for size in self.prefill_graph_sizes if size >= num_tokens),
            None,
        )
        if graph_size is None or graph_size not in self.prefill_graphs:
            return self.model(input_ids, positions)

        entries = self.prefill_graphs[graph_size]
        hidden = self.model.model.embed_tokens(input_ids)
        buffers = self.piecewise_buffers
        # Graphs share fixed addresses and replay serially on the compute
        # stream. Post writes directly into the next layer's pre inputs.
        buffers["hidden"][:num_tokens].copy_(hidden)
        buffers["hidden"][num_tokens:graph_size].zero_()
        buffers["attention"][num_tokens:graph_size].zero_()
        for layer_idx, layer in enumerate(self.model.model.layers):
            entry = entries[layer_idx]

            if context.target_features is not None and layer_idx in context.feature_layers:
                value = buffers["hidden"][:num_tokens]
                context.target_features[layer_idx] = (value.clone() if layer_idx == 0 else
                                                      value + buffers["residual"][:num_tokens])
            pre = entry["pre"]
            attention_pre, _ = pre["wrapper"](
                pre["hidden_in"], pre["residual_in"], graph_key=graph_size)
            attention_pre = tuple(
                tensor[:num_tokens] if tensor.dim() == 2 else tensor[:, :num_tokens]
                for tensor in attention_pre
            )
            attention_output = layer.forward_attention_core(
                attention_pre,
                positions,
                context.prefill_slices,
            )

            post = entry["post"]
            post["hidden_in"][:num_tokens].copy_(attention_output)
            post["wrapper"](post["hidden_in"], post["residual_in"], graph_key=graph_size)

        hidden, _ = self.model.model.norm(
            buffers["hidden"][:num_tokens],
            buffers["residual"][:num_tokens],
        )
        if context.target_features is not None:
            context.target_features["final"] = hidden
        return hidden

    def run_decode(self, input_ids, positions):
        with graph_mode(CUDAGraphMode.FULL):
            hidden = self.full_wrapper(input_ids, positions, graph_key=input_ids.size(0))
        return self.model.compute_logits(hidden)

    def _prepare_full_inputs(self, bs, input_ids, positions):
        context = get_context()
        graph_vars = self.decode_graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"].fill_(-1)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = (
            context.block_tables
        )
        graph_vars["state_indices"].zero_()
        graph_vars["state_indices"][:bs] = context.state_indices

    def clear_prefill(self) -> None:
        self.prefill_graphs.clear()
        self.piecewise_buffers = None
        self.split_graph.clear()
        self.prefill_graph_pool = None

    def clear(self) -> None:
        self.prefill_graphs.clear()
        self.piecewise_buffers = None
        self.split_graph.clear()
        self.prefill_graph_pool = None
        self.full_wrapper.clear()
        self.decode_graphs.clear()
        self.decode_graph_vars = None
        self.decode_graph_pool = None

    def _allocate_piecewise_buffers(
        self,
        max_tokens: int,
        hidden_size: int,
    ) -> dict:
        model_weight = self.model.model.embed_tokens.weight
        shape = (max_tokens, hidden_size)
        return {
            name: torch.zeros(shape, dtype=model_weight.dtype, device=model_weight.device)
            for name in ("hidden", "residual", "attention")
        } | {"pre_out": {}}

    def _piecewise_callable(self, layer, kind):
        return self.split_graph.backend.compile(layer, kind)

    def _allocate_pre_out(
        self,
        layer_idx: int,
        outputs: tuple[torch.Tensor, ...],
    ) -> list[torch.Tensor]:
        # Eager attention consumes projections before the next pre graph.
        # Equal layouts can share outputs across layers and token buckets.
        key = tuple((tuple(output.shape[1:]), output.dtype, output.device) for output in outputs)
        if key not in self.piecewise_buffers["pre_out"]:
            self.piecewise_buffers["pre_out"][key] = [
                output.new_zeros(
                    (max(self.prefill_graph_sizes), *output.shape[1:]),
                    device=output.device,
                )
                for output in outputs
            ]
        return self.piecewise_buffers["pre_out"][key]

    def _copy_pre_outputs(self, layer_idx: int, outputs, residual):
        static_outputs = self._allocate_pre_out(layer_idx, outputs)
        for dst, src in zip(static_outputs, outputs):
            dst[:src.shape[0]].copy_(src)
        static_residual = self.piecewise_buffers["residual"]
        static_residual[:residual.shape[0]].copy_(residual)
        return static_outputs, static_residual

    def _copy_post_outputs(self, outputs):
        hidden, residual = outputs
        self.piecewise_buffers["hidden"][:hidden.shape[0]].copy_(hidden)
        self.piecewise_buffers["residual"][:residual.shape[0]].copy_(
            residual
        )
        return (
            self.piecewise_buffers["hidden"],
            self.piecewise_buffers["residual"],
        )

    @torch.inference_mode()
    def capture_prefill_layer_segments(
        self,
        layer,
        graph_size: int,
        has_residual: bool,
    ):
        buffers = self.piecewise_buffers
        layer_idx = next(
            index
            for index, item in enumerate(self.model.model.layers)
            if item is layer
        )
        pre_hidden = buffers["hidden"][:graph_size]
        pre_residual = (
            buffers["residual"][:graph_size] if has_residual else None
        )
        pre_wrapper = self.split_graph.piece(layer, "pre")
        pre_fn = pre_wrapper.runnable
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                pre_outputs, _ = pre_fn(pre_hidden, pre_residual)
        self._allocate_pre_out(layer_idx, pre_outputs)
        torch.cuda.synchronize()

        with graph_mode(CUDAGraphMode.PIECEWISE):
            pre_graph, (static_pre_outputs, static_pre_residual) = pre_wrapper.capture(
                graph_size, (pre_hidden, pre_residual), self.prefill_graph_pool,
                transform=lambda output: self._copy_pre_outputs(layer_idx, *output),
            )
        torch.cuda.synchronize()

        post_hidden = buffers["attention"][:graph_size]
        post_residual = buffers["residual"][:graph_size]
        post_wrapper = self.split_graph.piece(layer, "post")
        post_fn = post_wrapper.runnable
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                post_outputs = post_fn(post_hidden, post_residual)
        torch.cuda.synchronize()

        with graph_mode(CUDAGraphMode.PIECEWISE):
            post_graph, (static_post_hidden, static_post_residual) = post_wrapper.capture(
                graph_size, (post_hidden, post_residual), self.prefill_graph_pool,
                transform=self._copy_post_outputs,
            )
        torch.cuda.synchronize()

        return {
            "pre": {
                "graph": pre_graph,
                "wrapper": pre_wrapper,
                "hidden_in": pre_hidden,
                "residual_in": pre_residual,
                "outputs": static_pre_outputs,
                "residual_out": static_pre_residual,
            },
            "post": {
                "graph": post_graph,
                "wrapper": post_wrapper,
                "hidden_in": post_hidden,
                "residual_in": post_residual,
                "hidden_out": static_post_hidden,
                "residual_out": static_post_residual,
            },
        }
