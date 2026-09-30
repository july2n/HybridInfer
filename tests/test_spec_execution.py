from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch
from hybridinfer.engine.sequence import Sequence
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode.config import SpeculativeConfig
from hybridinfer.scheduler import Scheduler
from hybridinfer.spec_decode.ngram import NgramProposer


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old

    def scheduler(self, **kwargs):
        values = dict(max_num_seqs=2, max_model_len=24, max_num_batched_tokens=8,
                      eos=-1, kvcache_block_size=4, num_kvcache_blocks=16,
                      enable_prefix_cache=True, is_hybrid=False,
                      speculative=SpeculativeConfig(enabled=True, ngram_max=2))
        values.update(kwargs)
        return Scheduler(SimpleNamespace(**values))

    def ready(self, scheduler):
        seq = Sequence([1, 2, 3, 4, 1, 2], SamplingParams(temperature=0, max_tokens=10))
        scheduler.block_manager.allocate(seq, 0)
        seq.num_cached_tokens = 5
        seq.is_prefill = False
        scheduler.running.append(seq)
        scheduler.resident.add(seq.seq_id)
        return seq

    def test_commit_publishes_only_computed_confirmed_pages(self):
        scheduler = self.scheduler()
        seq = self.ready(scheduler)
        selected, plans = scheduler.begin_speculative_batch()
        speculative_seq, plan = selected[0], plans[0]
        self.assertIs(speculative_seq, seq)
        self.assertEqual(plan.candidates, (3, 4, 1, 2))
        self.assertIn(seq.seq_id, scheduler.in_flight)
        self.assertEqual(list(scheduler.running), [])
        from hybridinfer.sampling.greedy_reference import accept_greedy
        result = accept_greedy(plan, (3, 4, 99, 99, 99), remaining_output_tokens=10, max_model_len=24)
        scheduler.finish_speculative(seq, result)
        self.assertEqual(seq.num_cached_tokens, 8)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4, 1, 2, 3, 4, 99])
        self.assertEqual(len(seq.block_table), 2)
        block = scheduler.block_manager.blocks[seq.block_table[1]]
        self.assertEqual(block.token_ids, [1, 2, 3, 4])
        self.assertNotIn(99, block.token_ids)
        self.assertEqual(scheduler.in_flight, set())

    def test_abort_is_schedulable_and_recovers_pages(self):
        scheduler = self.scheduler()
        seq = self.ready(scheduler)
        before = len(scheduler.block_manager.free_block_ids)
        scheduler.begin_speculative_batch()
        scheduler.abort_speculative_batch([seq])
        self.assertEqual(len(scheduler.block_manager.free_block_ids), before)
        self.assertEqual(list(scheduler.running), [seq])
        self.assertFalse(scheduler.in_flight)

    def test_next_round_draft_is_consumed_once_and_refreshed_after_rejection(self):
        scheduler = self.scheduler()
        seq = self.ready(scheduler)
        scheduler.draft_proposer = Mock(wraps=NgramProposer(scheduler.speculative))
        scheduler.prepare_next_draft(seq)
        cached = seq.spec_token_ids
        self.assertTrue(cached)
        self.assertEqual(scheduler.draft_proposer.propose.call_count, 1)

        selected, plans = scheduler.begin_speculative_batch()
        self.assertEqual(selected, [seq])
        self.assertEqual(plans[0].candidates, cached)
        self.assertEqual(scheduler.draft_proposer.propose.call_count, 1)

        from hybridinfer.sampling.greedy_reference import accept_greedy
        result = accept_greedy(plans[0], (99,) * (len(cached) + 1),
                               remaining_output_tokens=10, max_model_len=24)
        scheduler.finish_speculative(seq, result)
        self.assertEqual(seq.token_ids[-1], 99)
        self.assertEqual(seq.spec_base_length, seq.num_cached_tokens)
        self.assertEqual(scheduler.draft_proposer.propose.call_count, 2)

    def test_disabled_random_batch_and_resource_fallbacks(self):
        for kwargs, reason in (({'speculative': None}, None), ({'num_kvcache_blocks': 2}, 'kv_capacity_or_shared_tail')):
            scheduler = self.scheduler(**kwargs)
            seq = self.ready(scheduler)
            self.assertIsNone(scheduler.begin_speculative_batch())
            self.assertEqual(list(scheduler.running), [seq])
            if reason:
                self.assertEqual(scheduler.spec_fallbacks[reason], 1)
        scheduler = self.scheduler()
        seq = self.ready(scheduler)
        seq.temperature = 1
        self.assertIsNone(scheduler.begin_speculative_batch())
        self.assertEqual(scheduler.spec_fallbacks['temperature'], 1)
        seq.temperature = 0
        scheduler.in_flight.add(123)
        self.assertIsNone(scheduler.begin_speculative_batch())
        self.assertEqual(scheduler.spec_fallbacks['batch_or_prefill'], 1)


class RoutingTests(unittest.TestCase):
    def test_spec_verification_projects_all_rows_and_bypasses_decode_graphs(self):
        from hybridinfer.engine.model_runner import ModelRunner
        from hybridinfer.utils.context import set_context, reset_context, BatchDescriptor
        runner = ModelRunner.__new__(ModelRunner)
        runner.enforce_eager = False
        runner.cuda_graphs = SimpleNamespace(decode_graphs={1: object()},
                                             run_decode=lambda *args: self.fail('decode graph used'))
        runner.model = lambda ids, positions: ids[:, None].float()
        runner.model.compute_logits = lambda hidden: hidden
        runner._sample_indices = torch.tensor([2])
        set_context(True, batch_descriptor=BatchDescriptor(mode='spec_decode'))
        try:
            actual = runner.run_model(torch.tensor([1, 2, 3]), torch.arange(3), True)
            self.assertEqual(actual[:, 0].tolist(), [1, 2, 3])
        finally:
            reset_context()
