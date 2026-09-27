"""Hybrid prefix ownership and scheduler regressions, runnable without CUDA."""
from types import SimpleNamespace
import unittest

from hybridinfer.engine.sequence import Sequence
from hybridinfer.engine.block_manager import BlockManager
from hybridinfer.engine.prefix_checkpoint import PrefixCheckpointManager
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.scheduler import Scheduler


class PrefixCacheTests(unittest.TestCase):
    def setUp(self):
        self.old_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_size

    def scheduler(self, **overrides):
        values = dict(max_num_seqs=3, max_model_len=64, kvcache_block_size=4,
                      max_num_batched_tokens=64, num_kvcache_blocks=32,
                      enable_prefix_cache=True, is_hybrid=True,
                      prefix_cache_num_snapshots=2, eos=-1)
        values.update(overrides)
        return Scheduler(SimpleNamespace(**values))

    def seq(self, tokens):
        return Sequence(tokens, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True))

    def finish(self, scheduler):
        while not scheduler.is_finished():
            seqs, prefill = scheduler.schedule()
            self.assertTrue(seqs)
            scheduler.postprocess(seqs, [30] * len(seqs), prefill)

    def test_unpublished_and_pinned_checkpoints_cannot_be_reused_or_evicted(self):
        pool = PrefixCheckpointManager(1)
        a, b = self.seq(list(range(9))), self.seq(list(range(10, 19)))
        snapshot = pool.reserve(a, 8)
        self.assertEqual(pool.lookup(a, 2), (0, None))
        self.assertIsNone(pool.reserve(b, 8))
        pool.release(snapshot, publish=True)
        self.assertEqual(pool.lookup(a, 2), (2, snapshot))
        self.assertIsNone(pool.reserve(b, 8))
        pool.release(snapshot)
        self.assertEqual(pool.reserve(b, 8), snapshot)
        self.assertEqual(pool.lookup(a, 2), (0, None))

    def test_tail_split_and_exact_match_keeps_logits_token(self):
        scheduler = self.scheduler()
        a = self.seq(list(range(8)))
        scheduler.add(a)
        batch, prefill = scheduler.schedule()
        self.assertEqual(a.num_scheduled_tokens, 4)
        self.assertIsNotNone(a.save_snapshot_id)
        scheduler.postprocess(batch, [30], prefill)
        self.finish(scheduler)
        b = self.seq(list(range(8)))
        scheduler.add(b)
        batch, prefill = scheduler.schedule()
        self.assertEqual(b.num_cached_tokens, 4)
        self.assertEqual(b.num_scheduled_tokens, 4)
        self.assertIsNotNone(b.restore_snapshot_id)
        scheduler.postprocess(batch, [30], prefill)
        self.assertEqual(scheduler.checkpoints.hit_tokens, 4)

    def test_concurrent_restores_pin_shared_snapshot(self):
        scheduler = self.scheduler()
        scheduler.add(self.seq(list(range(9))))
        self.finish(scheduler)
        b, c = self.seq(list(range(8)) + [20]), self.seq(list(range(8)) + [21])
        scheduler.add(b)
        scheduler.add(c)
        batch, prefill = scheduler.schedule()
        self.assertEqual(len(batch), 2)
        self.assertEqual(b.restore_snapshot_id, c.restore_snapshot_id)
        entry = next(iter(scheduler.checkpoints.entries.values()))
        self.assertEqual(entry.pins, 2)
        scheduler.postprocess(batch, [30, 30], prefill)
        self.assertEqual(entry.pins, 0)
        self.assertEqual(scheduler.checkpoints.hits, 2)

    def test_kv_eviction_invalidates_otherwise_ready_snapshot(self):
        scheduler = self.scheduler(num_kvcache_blocks=3)
        a = self.seq(list(range(9)))
        scheduler.add(a)
        self.finish(scheduler)
        self.assertTrue(any(e.ready for e in scheduler.checkpoints.entries.values()))
        bm = scheduler.block_manager
        # Consume every free physical block, overwriting the cached payloads.
        ids = [bm._allocate_block() for _ in range(3)]
        for block_id in ids:
            bm.blocks[block_id].ref_count = 0
            bm._deallocate_block(block_id)
        b = self.seq(list(range(9)))
        scheduler.add(b)
        scheduler.schedule()
        self.assertEqual(b.num_cached_tokens, 0)
        self.assertIsNone(b.restore_snapshot_id)

    def test_snapshot_eviction_falls_back_despite_kv_hit(self):
        scheduler = self.scheduler(prefix_cache_num_snapshots=1)
        scheduler.add(self.seq(list(range(9))))
        self.finish(scheduler)
        scheduler.add(self.seq(list(range(10, 19))))
        self.finish(scheduler)
        a = self.seq(list(range(9)))
        self.assertEqual(scheduler.block_manager.find_cached_blocks(a), 2)
        scheduler.add(a)
        scheduler.schedule()
        self.assertEqual(a.num_cached_tokens, 0)

    def test_failed_admission_releases_reader_pin(self):
        scheduler = self.scheduler(num_kvcache_blocks=3)
        scheduler.add(self.seq(list(range(9))))
        self.finish(scheduler)
        # Cached two-block prefix plus two new blocks cannot fit in three.
        scheduler.add(self.seq(list(range(8)) + [40] * 5))
        with self.assertRaises(RuntimeError):
            scheduler.schedule()
        self.assertTrue(all(e.pins == 0 for e in scheduler.checkpoints.entries.values()))

    def test_disabled_cache_does_not_reuse_existing_kv(self):
        scheduler = self.scheduler(enable_prefix_cache=False)
        a = self.seq(list(range(9)))
        scheduler.add(a)
        batch, prefill = scheduler.schedule()
        scheduler.block_manager.hash_blocks(a)
        scheduler.postprocess(batch, [30], prefill)
        b = self.seq(list(range(9)))
        self.assertGreater(scheduler.block_manager.find_cached_blocks(b), 0)
        scheduler.add(b)
        scheduler.schedule()
        self.assertEqual(b.num_cached_tokens, 0)

    def test_subblock_budget_eventually_publishes_boundary(self):
        scheduler = self.scheduler(max_num_batched_tokens=3)
        a = self.seq(list(range(9)))
        scheduler.add(a)
        self.finish(scheduler)
        self.assertTrue(any(len(e.tokens) == 8 and e.ready
                            for e in scheduler.checkpoints.entries.values()))

    def test_pending_batch_checkpoint_is_invisible_to_late_arrival(self):
        scheduler = self.scheduler(max_num_batched_tokens=8)
        a, b = self.seq(list(range(9))), self.seq(list(range(9)))
        scheduler.add(a)
        first, prefill = scheduler.schedule()
        scheduler.add(b)
        second, second_prefill = scheduler.schedule()
        self.assertEqual(b.num_cached_tokens, 0)
        self.assertIsNone(b.restore_snapshot_id)
        self.assertIsNone(b.save_snapshot_id)  # same prefix write is pending
        scheduler.postprocess(first, [30], prefill)
        scheduler.postprocess(second, [30], second_prefill)
        self.finish(scheduler)
        self.assertTrue(all(e.pins == 0 for e in scheduler.checkpoints.entries.values()))

    def test_budget_deferral_releases_reader_pin(self):
        scheduler = self.scheduler(max_num_batched_tokens=8)
        scheduler.add(self.seq(list(range(9))))
        self.finish(scheduler)
        scheduler.add(self.seq([50]))
        b = self.seq(list(range(8)) + [60] * 8)
        scheduler.add(b)
        batch, _ = scheduler.schedule()
        self.assertEqual(len(batch), 1)
        self.assertIsNone(b.restore_snapshot_id)
        self.assertTrue(all(e.pins == 0 for e in scheduler.checkpoints.entries.values()))

    def test_pinned_pool_does_not_force_extra_prefill_chunk(self):
        scheduler = self.scheduler(prefix_cache_num_snapshots=1)
        owner = self.seq(list(range(9)))
        scheduler.checkpoints.reserve(owner, 8)
        other = self.seq(list(range(10, 19)))
        scheduler.add(other)
        scheduler.schedule()
        self.assertEqual(other.num_scheduled_tokens, 9)
        self.assertIsNone(other.save_snapshot_id)

    def test_existing_snapshot_with_evicted_kv_does_not_force_split(self):
        scheduler = self.scheduler()
        a = self.seq(list(range(9)))
        snapshot_id = scheduler.checkpoints.reserve(a, 8)
        scheduler.checkpoints.release(snapshot_id, publish=True)
        scheduler.add(a)  # No KV was stored; existing snapshot alone cannot hit.
        scheduler.schedule()
        self.assertEqual(a.num_cached_tokens, 0)
        self.assertEqual(a.num_scheduled_tokens, 9)
        self.assertIsNone(a.save_snapshot_id)

    def test_state_payload_restores_without_aliasing(self):
        import torch
        from hybridinfer.engine.model_runner import ModelRunner

        layers = [SimpleNamespace(conv_states=torch.randn(2, 6, 3),
                                  recurrent_states=torch.randn(2, 2, 4, 4))
                  for _ in range(2)]
        runner = SimpleNamespace(config=SimpleNamespace(enable_prefix_cache=True,
                                                         prefix_cache_num_snapshots=1),
                                 gdn_layers=layers)
        ModelRunner.allocate_prefix_snapshots(runner)
        a = self.seq([1])
        a.save_snapshot_id = 0
        ModelRunner.copy_prefix_state(runner, a, 0, restore=False)
        a.restore_snapshot_id = 0
        ModelRunner.copy_prefix_state(runner, a, 1, restore=True)
        for layer, (conv, recurrent) in zip(layers, runner.prefix_snapshots):
            self.assertTrue(torch.equal(layer.conv_states[1], conv[0]))
            self.assertTrue(torch.equal(layer.recurrent_states[1], recurrent[0]))
            layer.conv_states[1].add_(10)
            layer.recurrent_states[1].zero_()
            self.assertTrue(torch.equal(layer.conv_states[0], conv[0]))
            self.assertTrue(torch.equal(layer.recurrent_states[0], recurrent[0]))


if __name__ == '__main__':
    unittest.main()
