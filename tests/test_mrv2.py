"""MRV2 state/stream regressions; no checkpoint or attention kernels required.

PYTHONPATH=src python -m unittest discover -s tests -v
"""
import pickle
from types import SimpleNamespace
import unittest

import torch

from hybridinfer.engine.sequence import Sequence
from hybridinfer.engine.request_state import InputBatch, RequestState
from hybridinfer.engine.staged_write import StagedWriteTensor, to_device
from hybridinfer.engine.input_prep import prepare_inputs, advance, commit_sampled
from hybridinfer.engine.async_output import AsyncModelOutput
from hybridinfer.layers.sampler import Sampler
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.scheduler import Scheduler


def config(**kwargs):
    values = dict(max_num_seqs=3, max_model_len=16, kvcache_block_size=4,
                  max_num_batched_tokens=8, num_kvcache_blocks=32,
                  enable_prefix_cache=False, eos=-1)
    values.update(kwargs)
    return SimpleNamespace(**values)


def seq(tokens, count=None, blocks=()):
    result = Sequence(tokens, SamplingParams(temperature=0, max_tokens=5, seed=37))
    result.num_scheduled_tokens = len(tokens) if count is None else count
    result.block_table = list(blocks)
    return result


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_pickle_preserves_tp_request_state(self):
        a = seq([2, 3], 1, [7])
        a.is_prefill = False
        b = pickle.loads(pickle.dumps(a))
        self.assertEqual(a.__dict__, b.__dict__)

    def test_stable_slots_and_atomic_capacity_failure(self):
        batch = InputBatch(2)
        a, b, c = seq([1]), seq([2]), seq([3])
        batch.update([a, b])
        self.assertEqual(batch.slots_for([b, a]), [1, 0])
        with self.assertRaises(RuntimeError):
            batch.update([c])
        self.assertEqual(batch.seq_id_to_slot, {a.seq_id: 0, b.seq_id: 1})
        batch.remove(a.seq_id)
        batch.update([c])
        self.assertEqual(batch.slots_for([c, b]), [0, 1])

    def test_total_resident_limit_across_batches(self):
        scheduler = Scheduler(config(max_num_seqs=1))
        a, b = seq([1]), seq([2])
        scheduler.add(a)
        scheduler.add(b)
        first, prefill = scheduler.schedule()
        self.assertEqual(first, [a])
        self.assertEqual(scheduler.schedule()[0], [])
        scheduler.postprocess(first, [3], prefill)
        self.assertEqual(scheduler.schedule()[0], [a])

    def test_preemption_releases_slot_and_replays_tokens(self):
        scheduler = Scheduler(config())
        a = seq([1, 2])
        scheduler.add(a)
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [3], prefill)
        scheduler.running.remove(a)
        scheduler.preempt(a)
        self.assertEqual(scheduler.preempted, [a.seq_id])
        self.assertNotIn(a.seq_id, scheduler.resident)
        self.assertEqual(a.num_cached_tokens, 0)
        batch, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(a.num_scheduled_tokens, 3)

    def test_chunk_continuation_bypasses_blocked_admission(self):
        scheduler = Scheduler(config(max_num_seqs=2, max_num_batched_tokens=3,
                                     num_kvcache_blocks=2))
        a, b = seq([1, 2, 3, 4, 5]), seq([7])
        scheduler.add(a)
        scheduler.add(b)
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [9], prefill)
        self.assertEqual(list(scheduler.waiting), [b, a])
        batch, prefill = scheduler.schedule()
        self.assertEqual(batch, [a])
        self.assertEqual(a.num_scheduled_tokens, 2)

    def test_length_limit_finishes_without_out_of_bounds_decode(self):
        scheduler = Scheduler(config(max_model_len=2))
        a = seq([1, 2])
        scheduler.add(a)
        batch, prefill = scheduler.schedule()
        scheduler.postprocess(batch, [3], prefill)
        self.assertTrue(a.is_finished)
        self.assertFalse(scheduler.resident)
        self.assertTrue(scheduler.is_finished())
        with self.assertRaises(ValueError):
            scheduler.add(seq([1, 2, 3]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GPUTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_incremental_writes_are_ordered_and_do_not_alias_host(self):
        state = StagedWriteTensor((2, 5), torch.int32)
        state.stage_write(0, 1, [3, 4])
        state.stage_write(0, 2, [9])
        state.apply_write()
        state.stage_write(1, 0, [7])
        state.apply_write()
        self.assertEqual(state.tensor.tolist(), [[0, 3, 9, 0, 0], [7, 0, 0, 0, 0]])

    def test_mixed_inputs_chunk_progress_and_sample_commit(self):
        state = RequestState(config())
        a = seq([10, 11, 12, 13, 14], 3, [2, 7])
        b = seq([20, 21], 2, [4])
        batch = InputBatch(3)
        _, new = batch.update([a, b])
        state.update([a, b], [0, 1], new)
        slots = to_device([1, 0], torch.int64, 'cuda')
        counts = to_device([2, 3], torch.int32, 'cuda')
        offsets = to_device([0, 2, 5], torch.int32, 'cuda')
        ids, positions, mapping, lengths, cu_k, tables = prepare_inputs(state, slots, offsets, counts, 5, 3, 4)
        self.assertEqual(ids.tolist(), [20, 21, 10, 11, 12])
        self.assertEqual(positions.tolist(), [0, 1, 0, 1, 2])
        self.assertEqual(mapping.tolist(), [16, 17, 8, 9, 10])
        self.assertEqual(cu_k.tolist(), [0, 2, 5])
        advance(state, slots, counts)
        last = torch.zeros(3, dtype=torch.int64, device='cuda')
        commit_sampled(state, slots, to_device([22, 99], torch.int64, 'cuda'),
                       to_device([1, 0], torch.int32, 'cuda'), last)
        self.assertEqual(state.tokens.tensor[0, :5].tolist(), [10, 11, 12, 13, 14])
        self.assertEqual(state.tokens.tensor[1, :3].tolist(), [20, 21, 22])
        counts = to_device([2, 1], torch.int32, 'cuda')
        slots = to_device([0, 1], torch.int64, 'cuda')
        offsets = to_device([0, 2, 3], torch.int32, 'cuda')
        ids, positions, mapping, lengths, cu_k, tables = prepare_inputs(state, slots, offsets, counts, 3, 2, 4)
        self.assertEqual(ids.tolist(), [13, 14, 22])
        self.assertEqual(positions.tolist(), [3, 4, 2])
        self.assertEqual(mapping.tolist(), [11, 28, 18])
        self.assertEqual(lengths.tolist(), [5, 3])
        self.assertEqual(cu_k.tolist(), [0, 5, 8])

    def test_reused_slot_clears_stale_blocks(self):
        state = RequestState(config())
        a = seq([1, 2, 3, 4, 5], blocks=[8, 9])
        state.update([a], [0], [(a, 0)])
        b = seq([7], blocks=[3])
        state.update([b], [0], [(b, 0)])
        self.assertEqual(state.block_tables.tensor[0].tolist(), [3, -1, -1, -1])
        self.assertEqual(state.computed.tensor[0].item(), 0)
        self.assertEqual(state.tokens.tensor[0, 0].item(), 7)

    def test_sampler_greedy_ties_large_vocab_and_noncontiguous_rows(self):
        sampler = Sampler()
        logits = torch.full((3, 4099), -float('inf'), device='cuda')
        logits[0, 1025] = 3
        logits[0, 3000] = 3
        logits[1, 4098] = 5
        logits[2, 17] = 1
        result = sampler(logits[::2], torch.zeros(2, device='cuda'))
        self.assertEqual(result.tolist(), [1025, 17])

    def test_sampling_distribution_and_seeded_reordering(self):
        sampler = Sampler()
        n = 12000
        logits = torch.tensor([0., 1., 2.], device='cuda').expand(n, -1)
        slots = torch.arange(n, device='cuda')
        temps = torch.ones(n, device='cuda')
        seeds = torch.arange(n, device='cuda', dtype=torch.int64) + 713
        positions = torch.full((n,), 5, device='cuda', dtype=torch.int32)
        output = sampler.sample(logits, slots, temps, seeds, positions)
        actual = torch.bincount(output, minlength=3).float() / n
        self.assertTrue(torch.allclose(actual, logits[0].softmax(0), atol=.02, rtol=0))
        reverse = slots.flip(0)
        other = sampler.sample(logits, reverse, temps, seeds, positions)
        self.assertTrue(torch.equal(other, output.flip(0)))

    def test_unconsumed_output_owns_memory(self):
        stream = torch.cuda.Stream()
        handles = []
        for i in range(7):
            gpu = torch.full((4,), i, dtype=torch.int64, device='cuda')
            host = torch.empty(4, dtype=torch.int64, pin_memory=True)
            event = torch.cuda.Event()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                host.copy_(gpu, non_blocking=True)
                gpu.record_stream(stream)
                event.record(stream)
            handles.append(AsyncModelOutput(gpu, host, event))
        for i, handle in enumerate(handles):
            self.assertEqual(handle.get_output(), [i] * 4)
            self.assertEqual(handle.get_output(), [i] * 4)


class ToyModel(torch.nn.Module):
    """Deterministic per-position logits for testing the real runner loop."""
    def forward(self, ids, positions):
        target = (ids + positions + 1) % 32
        logits = torch.full((ids.numel(), 32), -20., device=ids.device)
        return logits.scatter_(1, target[:, None], 20.)

    def compute_logits(self, hidden):
        return hidden


def make_runner(cfg, graphs=False):
    from hybridinfer.engine.model_runner import ModelRunner
    from hybridinfer.engine.cuda_graph import CudaGraphManager
    runner = ModelRunner.__new__(ModelRunner)
    cfg.hf_config = SimpleNamespace(hidden_size=32)
    runner.config = cfg
    runner.rank = 0
    runner.world_size = 1
    runner.block_size = cfg.kvcache_block_size
    runner.enforce_eager = not graphs
    runner.use_prefill_cudagraph = False
    runner.async_output = True
    runner.model = ToyModel()
    runner.gdn_layers = []
    from hybridinfer.engine.kv_cache_manager import KVCacheStorage
    runner.kv_cache_manager = KVCacheStorage(cfg, runner.model, [])
    runner.sampler = Sampler()
    runner.output_copy_stream = torch.cuda.Stream()
    runner.sampled_token_ids_gpu = torch.empty(cfg.max_num_seqs, dtype=torch.int64, device='cuda')
    runner.batch_slots_gpu = torch.empty_like(runner.sampled_token_ids_gpu)
    runner.request_state = RequestState(cfg)
    runner.input_batch = InputBatch(cfg.max_num_seqs)
    runner._pending = None
    runner.cuda_graphs = CudaGraphManager(runner)
    if graphs:
        runner.cuda_graphs.capture()
    return runner


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_small_decode_batches_use_exact_graph_without_padding(self):
        runner = make_runner(config(max_num_seqs=8), graphs=True)
        self.assertEqual(runner.cuda_graphs.decode_graph_sizes, list(range(1, 9)))
        seqs = [seq([3, 4], 2, [i]) for i in range(3)]
        runner.execute_model(seqs, True)
        tokens = runner.sample_tokens().get_output()
        for item, token in zip(seqs, tokens):
            item.append_token(token)
            item.num_cached_tokens = 2
            item.num_scheduled_tokens = 1
            item.is_prefill = False
        graph = runner.cuda_graphs.decode_graphs[3]
        replayed = []
        runner.cuda_graphs.decode_graphs[3] = SimpleNamespace(
            replay=lambda: (replayed.append(True), graph.replay()))
        runner.execute_model(seqs, False)
        actual = runner.sample_tokens().get_output()
        self.assertEqual(replayed, [True])
        self.assertEqual(actual, [(token + 3) % 32 for token in tokens])
        self.assertEqual(runner.request_state.computed.tensor[:3].tolist(), [3] * 3)

    def test_real_runner_chunked_prefill_to_graph_decode(self):
        runner = make_runner(config(), graphs=True)
        a = seq([3, 4, 5, 6], 2, [2, 3])
        runner.execute_model([a], True)
        with self.assertRaises(RuntimeError):
            runner.execute_model([a], True)
        runner.sample_tokens().get_output()
        # Intermediate chunk must not overwrite future prompt tokens.
        self.assertEqual(runner.request_state.tokens.tensor[0, :4].tolist(), [3, 4, 5, 6])
        a.num_cached_tokens = 2
        a.num_scheduled_tokens = 2
        runner.execute_model([a], True)
        token = runner.sample_tokens().get_output()[0]
        self.assertEqual(token, 10)
        a.append_token(token)
        a.num_cached_tokens = 4
        a.num_scheduled_tokens = 1
        a.is_prefill = False
        runner.execute_model([a], False)
        self.assertEqual(runner.sample_tokens().get_output(), [15])
        self.assertEqual(runner.request_state.computed.tensor[0].item(), 5)

    def test_engine_queue_matches_serial_under_slot_pressure(self):
        from collections import deque
        from hybridinfer.engine.llm_engine import LLMEngine
        def run(depth, graphs, blocks=32):
            cfg = config(max_num_seqs=2, max_model_len=32, max_num_batched_tokens=3,
                         num_kvcache_blocks=blocks)
            engine = LLMEngine.__new__(LLMEngine)
            engine.config = cfg
            engine.scheduler = Scheduler(cfg)
            engine.model_runner = make_runner(cfg, graphs)
            engine.max_concurrent_batches = depth
            engine.batch_queue = deque()
            engine._coalesce_armed = False
            engine._coalesce_before = 0
            engine._coalesce_source_ids = frozenset()
            engine.coalesce_stats = dict(coalesce_attempts=0, coalesce_successes=0,
                merged_batches=0, merged_requests=0, batch_size_before={}, batch_size_after={})
            requests = [seq([3, 4, 5, 6, 7]), seq([9]), seq([11, 12]), seq([20])]
            for request in requests:
                engine.scheduler.add(request)
            for _ in range(100):
                if engine.is_finished():
                    break
                engine.step()
            self.assertTrue(engine.is_finished())
            self.assertFalse(engine.model_runner.input_batch.seq_id_to_slot)
            return [s.completion_token_ids for s in requests]
        baseline = run(1, False)
        self.assertEqual(run(2, True), baseline)
        self.assertEqual(run(5, True), baseline)
        # Three blocks are the minimum to hold the longest request's final
        # forward (9 tokens). Other requests must be preempted to fit it.
        self.assertEqual(run(2, True, blocks=3), baseline)

    def test_runner_outputs_survive_multiple_unconsumed_batches(self):
        runner = make_runner(config(max_num_seqs=7))
        requests = [seq([i], blocks=[i]) for i in range(7)]
        outputs = []
        for request in requests:
            runner.execute_model([request], True)
            outputs.append(runner.sample_tokens())
        for i, output in enumerate(outputs):
            self.assertEqual(output.get_output(), [i + 1])


if __name__ == '__main__':
    unittest.main()
