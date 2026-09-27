from types import SimpleNamespace
import unittest
import torch

from hybridinfer.engine.sequence import Sequence
from hybridinfer.engine.request_state import InputBatch
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.spec_decode.execution import verify_sequential
from hybridinfer.spec_decode.config import SpeculativeConfig
from hybridinfer.scheduler import Scheduler
from hybridinfer.utils.context import get_context


def make_runner(seq, predictions, fail_replay=False):
    layer = SimpleNamespace(conv_states=torch.ones(3, 2), recurrent_states=torch.ones(3, 2))
    batch = InputBatch(2)
    batch.update([Sequence([9]), seq])
    tokens = torch.full((2, 25), -1, dtype=torch.int64)
    tokens[1, :seq.num_tokens] = torch.tensor(seq.token_ids)
    computed = torch.tensor([1, seq.num_cached_tokens], dtype=torch.int32)
    state = SimpleNamespace(tokens=SimpleNamespace(tensor=tokens), computed=SimpleNamespace(tensor=computed),
                            block_tables=SimpleNamespace(tensor=torch.tensor([[3, 4, 5], seq.block_table])),
                            update=lambda *args: None)
    kv = {}
    class Model:
        def __call__(self, ids, positions):
            context = get_context()
            slot = int(context.state_indices[0])
            position = int(positions[0])
            token = int(ids[0])
            for pool in (layer.conv_states, layer.recurrent_states):
                pool[slot].add_(token)
            if fail_replay and slot == 1:
                raise RuntimeError('replay failed')
            kv[position] = token
            return torch.tensor([[float(position)]])

        def compute_logits(self, hidden):
            row = int(hidden[0, 0]) - seq.num_cached_tokens
            logits = torch.full((1, 128), -100.)
            logits[0, predictions[row]] = 100.
            return logits
    return SimpleNamespace(_pending=None, world_size=1, input_batch=batch, request_state=state,
                           config=SimpleNamespace(max_num_seqs=2, max_model_len=24, eos=-1),
                           gdn_layers=[layer], block_size=4, model=Model(), kv=kv,
                           sampled_token_ids_gpu=torch.zeros(2, dtype=torch.int64),
                           spec_metrics=dict(rounds=0, draft_tokens=0, accepted_tokens=0,
                                             output_tokens=0, trial_tokens=0, replay_tokens=0,
                                             copy_seconds=0., verify_seconds=0., restore_seconds=0., commit_seconds=0.))


class ExecutionTests(unittest.TestCase):
    def test_each_acceptance_endpoint_and_truncated_endpoints(self):
        for accepted in range(4):
            for remaining in (1, 2, 10):
                seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0, max_tokens=remaining))
                seq.num_cached_tokens = 3
                seq.block_table = [0, 1, 2]
                predictions = [5, 6, 7, 8]
                if accepted < 3:
                    predictions[accepted] = 99
                runner = make_runner(seq, predictions)
                before = seq.token_ids.copy()
                r = verify_sequential(runner, seq, VerificationPlan(seq.seq_id, 3, 4, (5, 6, 7)))
                count = min(accepted+1, remaining)
                self.assertEqual(r.output_length, count)
                self.assertEqual(int(runner.request_state.computed.tensor[1]), 3+count)
                self.assertEqual(runner.request_state.tokens.tensor[1, 4:4+count].tolist(), list(r.token_ids))
                self.assertEqual(seq.token_ids, before)
                self.assertTrue(torch.equal(runner.gdn_layers[0].recurrent_states[0], torch.ones(2)))
                expected_inputs = [4, 5, 6, 7][:count]
                self.assertTrue(torch.equal(runner.gdn_layers[0].recurrent_states[1],
                                            torch.full((2,), float(1+sum(expected_inputs)))))
                self.assertEqual([runner.kv[i] for i in range(3, 3+count)], expected_inputs)
                self.assertEqual(runner.request_state.tokens.tensor[1, 4+count:].tolist(), [-1]*(21-count))
                self.assertFalse(get_context().state_indices is not None)

    def test_replay_failure_leaves_history_and_lengths_unchanged(self):
        seq = Sequence([1, 2, 3, 4], SamplingParams(temperature=0, max_tokens=10))
        seq.num_cached_tokens = 3
        seq.block_table = [0, 1, 2]
        runner = make_runner(seq, [99, 6, 7, 8], fail_replay=True)
        with self.assertRaises(RuntimeError):
            verify_sequential(runner, seq, VerificationPlan(seq.seq_id, 3, 4, (5, 6, 7)))
        self.assertTrue(torch.equal(runner.gdn_layers[0].recurrent_states[1], torch.ones(2)))
        self.assertEqual(int(runner.request_state.computed.tensor[1]), 3)
        self.assertEqual(seq.token_ids, [1, 2, 3, 4])
        self.assertEqual(runner.request_state.tokens.tensor[1, 4:].tolist(), [-1]*21)


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
        speculative_seq, plan = scheduler.begin_speculative()
        self.assertIs(speculative_seq, seq)
        self.assertEqual(plan.candidates, (3, 4, 1, 2))
        self.assertIn(seq.seq_id, scheduler.in_flight)
        self.assertEqual(list(scheduler.running), [])
        from hybridinfer.spec_decode.verifier import accept_greedy
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
        scheduler.begin_speculative()
        scheduler.abort_speculative(seq)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), before)
        self.assertEqual(list(scheduler.running), [seq])
        self.assertFalse(scheduler.in_flight)

    def test_disabled_random_batch_and_resource_fallbacks(self):
        for kwargs, reason in (({'speculative': None}, None), ({'num_kvcache_blocks': 2}, 'kv_capacity_or_shared_tail')):
            scheduler = self.scheduler(**kwargs)
            seq = self.ready(scheduler)
            self.assertIsNone(scheduler.begin_speculative())
            self.assertEqual(list(scheduler.running), [seq])
            if reason:
                self.assertEqual(scheduler.spec_fallbacks[reason], 1)
        scheduler = self.scheduler()
        seq = self.ready(scheduler)
        seq.temperature = 1
        self.assertIsNone(scheduler.begin_speculative())
        self.assertEqual(scheduler.spec_fallbacks['temperature'], 1)
        seq.temperature = 0
        scheduler.in_flight.add(123)
        self.assertIsNone(scheduler.begin_speculative())
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
