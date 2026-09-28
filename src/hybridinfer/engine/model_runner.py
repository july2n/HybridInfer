import pickle
import socket
import time

import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from hybridinfer.config import Config
from hybridinfer.engine.sequence import Sequence
from hybridinfer.engine.cuda_graph import CudaGraphManager
from hybridinfer.layers.sampler import Sampler
from hybridinfer.utils.context import set_context, get_context, reset_context, BatchDescriptor
from hybridinfer.utils.loader import load_model
from hybridinfer.utils.trace import trace_event

from .async_output import AsyncModelOutput
from .request_state import InputBatch, RequestState
from .staged_write import to_device
from .input_prep import prepare_inputs as prepare_gpu_inputs, advance, commit_sampled


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ModelRunner:

    def __init__(
        self,
        config: Config,
        rank: int,
        event: Event | list[Event],
        port: int | None = None,
        use_prefill_cudagraph: bool = True,
    ):
        from hybridinfer.layers.gated_delta_net import GatedDeltaNet
        from hybridinfer.models.qwen3_5 import Qwen3_5ForCausalLM

        self._closed = False
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Rendezvous on an ephemeral port: a fixed port lingers in TIME_WAIT
        # after exit and makes a rerun within ~60s fail with EADDRINUSE.
        if port is None:
            port = find_free_port()
        dist.init_process_group(
            "nccl",
            f"tcp://127.0.0.1:{port}",
            world_size=self.world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3_5ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.gdn_layers = [
            module
            for module in self.model.modules()
            if isinstance(module, GatedDeltaNet)
        ]
        self.sampler = Sampler()
        self.output_copy_stream = torch.cuda.Stream()
        # Benchmark can disable the copy stream to provide a synchronous D2H
        # baseline. Keep async output as the production default.
        self.async_output = True
        # The server benchmark can isolate prefill CUDA Graph overhead while
        # keeping decode graphs enabled. Production defaults to prefill graphs.
        self.use_prefill_cudagraph = use_prefill_cudagraph
        self.cuda_graphs = CudaGraphManager(self)
        self.sampled_token_ids_gpu = torch.empty(
            config.max_num_seqs, dtype=torch.int64, device="cuda",
        )
        # MRV2-style idx_mapping: batch_idx -> persistent request slot.
        self.batch_slots_gpu = torch.empty(
            config.max_num_seqs, dtype=torch.int64, device="cuda",
        )
        self.request_state = RequestState(config)
        self.input_batch = InputBatch(config.max_num_seqs)
        # Only the execute/sample entrypoints are paired; completed GPU
        # submissions may remain in flight independently in the engine queue.
        self._pending: tuple | None = None
        self.spec_metrics = dict(rounds=0, draft_tokens=0, accepted_tokens=0,
                                 output_tokens=0, trial_tokens=0, replay_tokens=0,
                                 copy_seconds=0., verify_seconds=0.,
                                 restore_seconds=0., commit_seconds=0.)
        self.allocate_gdn_state_pool()
        self.allocate_prefix_snapshots()
        self.warmup_model()
        self.input_batch.clear()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="hybridinfer", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="hybridinfer")
                self.loop()

    def exit(self):
        if self._closed:
            return
        self._closed = True
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

        # CUDA Graphs retain captured allocations through their Python
        # containers. Clear every runner-owned GPU reference, including the
        # lazily-created piecewise graphs, before the engine is discarded.
        self._pending = None
        self.cuda_graphs.clear()
        for name in ("gdn_layers",):
            value = getattr(self, name, None)
            if value is not None:
                value.clear()
        if hasattr(self, "input_batch"):
            self.input_batch.clear()
        self.model = None
        self.kv_cache = None
        self.prefix_snapshots = []
        self.sampled_token_ids_gpu = None
        self.batch_slots_gpu = None
        self.request_state = None
        self._step_counts = None
        self._sample_indices = None
        self._emit = None
        self.output_copy_stream = None
        torch.cuda.empty_cache()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        for event in self.event:
            while event.is_set():
                time.sleep(0.0001)
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise ValueError("TP command exceeds shared-memory capacity")
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.execute_model(seqs, True)
        async_output = self.sample_tokens()
        if async_output is not None:
            async_output.get_output()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # Only full_attention layers hold K/V cache; GDN layers keep their own
        # conv/recurrent state pools instead. Count cache-bearing modules from
        # the model structure (robust to both ``layer_types`` and the
        # ``full_attention_interval`` fallback, and to Dense checkpoints where
        # every layer is attention).
        num_attn_layers = sum(
            1 for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        )
        assert num_attn_layers > 0, "no attention layers found; cannot size KV cache"
        block_bytes = (
            2 * num_attn_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2, num_attn_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_attn_layers, (
            f"attention layer drift: assigned {layer_id} caches but sized "
            f"for {num_attn_layers}"
        )

    def allocate_gdn_state_pool(self):
        num_slots = self.config.max_num_seqs
        speculative = self.config.speculative
        if speculative and speculative.enabled:
            num_slots *= 2  # One private trial slot per active verification request.
        for layer in self.gdn_layers:
            layer.allocate_state_pool(num_slots)

    def allocate_prefix_snapshots(self):
        count = self.config.prefix_cache_num_snapshots if self.config.enable_prefix_cache else 0
        self.prefix_snapshots = [
            (layer.conv_states.new_empty((count, *layer.conv_states.shape[1:])),
             layer.recurrent_states.new_empty((count, *layer.recurrent_states.shape[1:])))
            for layer in self.gdn_layers
        ]

    def copy_prefix_state(self, seq, slot, restore):
        snapshot_id = seq.restore_snapshot_id if restore else seq.save_snapshot_id
        if snapshot_id is None:
            return
        for layer, (conv, recurrent) in zip(self.gdn_layers, self.prefix_snapshots):
            for runtime, cached in ((layer.conv_states, conv), (layer.recurrent_states, recurrent)):
                if restore:
                    runtime[slot].copy_(cached[snapshot_id])
                else:
                    cached[snapshot_id].copy_(runtime[slot])

    def prepare_inputs(self, seqs: list[Sequence], is_prefill: bool):
        counts_cpu = [seq.num_scheduled_tokens for seq in seqs]
        offsets_cpu = [0]
        for count in counts_cpu:
            offsets_cpu.append(offsets_cpu[-1] + count)
        device = self.batch_slots_gpu.device
        counts = to_device(counts_cpu, torch.int32, device)
        offsets = to_device(offsets_cpu, torch.int32, device)
        slots = self.batch_slots_gpu[:len(seqs)]
        ids, positions, mapping, lengths, cu_k, tables = prepare_gpu_inputs(
            self.request_state, slots, offsets, counts,
            offsets_cpu[-1], max(counts_cpu), self.block_size,
        )
        slices = list(zip(offsets_cpu[:-1], offsets_cpu[1:]))
        chunks = [(i, chunk) for i, count in enumerate(counts_cpu)
                  for chunk in range((count + 63) // 64)]
        # Attention's packed fresh-prefill fast path has no paged reads.
        paged = any(seq.num_cached_tokens > 0 for seq in seqs)
        block_tables = tables if not is_prefill or paged else None
        set_context(
            is_prefill,
            cu_seqlens_q=offsets if is_prefill else None,
            cu_seqlens_k=cu_k if is_prefill else None,
            max_seqlen_q=max(counts_cpu),
            max_seqlen_k=max(seq.num_cached_tokens + count for seq, count in zip(seqs, counts_cpu)),
            slot_mapping=mapping,
            context_lens=lengths if not is_prefill else None,
            block_tables=block_tables,
            state_indices=slots,
            prefill_slices=slices if is_prefill else None,
            prefill_chunk_indices=to_device(chunks, torch.int32, device) if is_prefill else None,
            batch_descriptor=BatchDescriptor(
                mode="prefill" if is_prefill else "decode",
                num_tokens=offsets_cpu[-1], num_reqs=len(seqs),
                uniform_token_count=None if is_prefill else 1,
                max_query_len=max(counts_cpu),
            ),
        )
        self._step_counts = counts
        self._sample_indices = offsets[1:].to(torch.int64) - 1
        self._emit = to_device([
            int(seq.num_cached_tokens + count >= seq.num_tokens)
            for seq, count in zip(seqs, counts_cpu)
        ], torch.int32, device)
        temperatures = self.request_state.temperatures.tensor.index_select(0, slots)
        return ids, positions, temperatures

    @torch.inference_mode()
    def compute_logits(self, hidden_states, is_prefill):
        # Project only the final scheduled token of each request. This avoids
        # allocating [all_prefill_tokens, vocab_size] logits for long prompts.
        if is_prefill:
            hidden_states = hidden_states.index_select(0, self._sample_indices)
        return self.model.compute_logits(hidden_states)

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ):
        context = get_context()
        descriptor = context.batch_descriptor
        if descriptor is not None:
            mode = descriptor.mode
        else:
            mode = "prefill" if is_prefill else "decode"

        if mode == "spec_decode":
            return self.model.compute_logits(self.model(input_ids, positions))

        if self.enforce_eager or not self.cuda_graphs.decode_graphs:
            return self.compute_logits(self.model(input_ids, positions), is_prefill)

        if mode == "prefill" and self.use_prefill_cudagraph:
            return self.cuda_graphs.run_prefill(input_ids, positions)
        if mode == "prefill":
            return self.compute_logits(self.model(input_ids, positions), True)

        # Ordinary decode dispatches through the graph manager.
        return self.cuda_graphs.run_decode(input_ids, positions)

    def verify_speculative(self, seq, plan):
        from hybridinfer.spec_decode.execution import verify_speculative
        return verify_speculative(self, seq, plan)

    def verify_speculative_batch(self, seqs, plans):
        from hybridinfer.spec_decode.batch_execution import verify_speculative_batch
        return verify_speculative_batch(self, seqs, plans)

    def execute_model(self, seqs: list[Sequence], is_prefill: bool) -> None:
        """MRV2 step: prepare inputs, enqueue the forward, return None.

        Only the kernels are queued onto the compute stream; the engine does
        not wait for them. Sampling is a separate call (sample_tokens), which
        reads the pending logits and performs the async D2H copy.
        """
        if self._pending is not None:
            raise RuntimeError("sample_tokens must consume the preceding execution")
        if not seqs:
            raise ValueError("cannot execute an empty batch")
        self.input_batch.remove_finished()
        seqs, new_entries = self.input_batch.update(seqs)
        slots = self.input_batch.slots_for(seqs)
        self.request_state.update(seqs, slots, new_entries)
        slots_t = to_device(slots, torch.int64, self.batch_slots_gpu.device)
        self.batch_slots_gpu[:len(seqs)].copy_(slots_t)

        if new_entries and self.gdn_layers:
            new_slots = to_device(
                [slot for _, slot in new_entries], torch.int64, self.batch_slots_gpu.device,
            )
            for layer in self.gdn_layers:
                layer.reset_state(new_slots)
            for seq, slot in new_entries:
                self.copy_prefix_state(seq, slot, restore=True)

        with trace_event(
            "prepare_inputs", "runner",
            args={"prefill": is_prefill, "bs": len(seqs)},
        ):
            input_ids, positions, temperatures = self.prepare_inputs(seqs, is_prefill)
        with trace_event(
            "run_model", "runner",
            args={"prefill": is_prefill, "tokens": input_ids.size(0)},
        ):
            logits = self.run_model(
                input_ids,
                positions,
                is_prefill,
            )
        for seq, slot in zip(seqs, slots):
            self.copy_prefix_state(seq, slot, restore=False)
        advance(self.request_state, self.batch_slots_gpu[:len(seqs)], self._step_counts)
        self._pending = (logits, temperatures, seqs, is_prefill)
        return None

    def sample_tokens(self) -> AsyncModelOutput | None:
        if self._pending is None:
            raise RuntimeError("sample_tokens requires a preceding execute_model")
        logits, temperatures, seqs, is_prefill = self._pending
        self._pending = None
        bs = len(seqs)
        slots = self.batch_slots_gpu[:bs]
        # Validation-only samplers may request row-aligned logits/GDN-state
        # diagnostics. The production sampler has no hooks, so its behavior
        # remains unchanged.
        set_batch_context = getattr(self.sampler, "set_batch_context", None)
        if set_batch_context is not None:
            set_batch_context(seqs, is_prefill)
        observe_gdn_state = getattr(self.sampler, "observe_gdn_state", None)
        if observe_gdn_state is not None:
            observe_gdn_state(self.gdn_layers, slots)

        if self.rank == 0:
            with trace_event("sampler", "runner", args={"prefill": is_prefill, "bs": bs}):
                if isinstance(self.sampler, Sampler):
                    state = self.request_state
                    token_ids = self.sampler.sample(
                        logits, slots, state.temperatures.tensor,
                        state.seeds.tensor, state.computed.tensor,
                    )
                else:
                    token_ids = self.sampler(logits, temperatures)
        else:
            token_ids = torch.empty(bs, dtype=torch.int64, device=slots.device)
        if self.world_size > 1:
            dist.broadcast(token_ids, src=0)
        commit_sampled(self.request_state, slots, token_ids, self._emit, self.sampled_token_ids_gpu)
        if self.rank != 0:
            reset_context()
            return None

        # Every output handle owns its host storage and event. Queue depth can
        # change without overwriting an older, unconsumed output.
        output_buf = torch.empty(bs, dtype=torch.int64, device="cpu", pin_memory=True)
        ready_event = torch.cuda.Event(blocking=True)

        if self.async_output:
            self.output_copy_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.output_copy_stream):
                # Non-blocking enqueue; the real copy latency surfaces as
                # d2h_wait on the consuming step.
                with trace_event("d2h_copy", "runner", args={"bs": bs}):
                    output_buf.copy_(token_ids, non_blocking=True)
                token_ids.record_stream(self.output_copy_stream)
                ready_event.record(self.output_copy_stream)
        else:
            # Blocking D2H baseline: return only after the CPU buffer is ready.
            with trace_event("d2h_copy_sync", "runner", args={"bs": bs}):
                output_buf.copy_(token_ids, non_blocking=False)
            ready_event.record(torch.cuda.current_stream())

        reset_context()
        return AsyncModelOutput(token_ids, output_buf, ready_event)

    def remove_request(self, seq_id: int):
        slot = self.input_batch.seq_id_to_slot.get(seq_id)
        if slot is not None:
            self.request_state.remove(slot)
        self.input_batch.remove(seq_id)

    def capture_cudagraph(self):
        """Capture decode and piecewise-prefill graphs through the manager."""
        self.cuda_graphs.capture()

    @property
    def graph_bs(self):
        return self.cuda_graphs.decode_graph_sizes

    @property
    def graphs(self):
        return self.cuda_graphs.decode_graphs

    @property
    def graph_vars(self):
        return self.cuda_graphs.decode_graph_vars

    @property
    def graph_pool(self):
        return self.cuda_graphs.decode_graph_pool

    @property
    def prefill_graph_sizes(self):
        return self.cuda_graphs.prefill_graph_sizes

    @prefill_graph_sizes.setter
    def prefill_graph_sizes(self, value):
        self.cuda_graphs.prefill_graph_sizes = list(value)

    @property
    def prefill_piecewise_graphs(self):
        return self.cuda_graphs.prefill_graphs
