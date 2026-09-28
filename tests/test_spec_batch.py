from itertools import product
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hybridinfer.engine.request_state import InputBatch
from hybridinfer.engine.sequence import Sequence
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.scheduler import Scheduler
from hybridinfer.spec_decode.batch_execution import verify_speculative_batch
from hybridinfer.spec_decode.batch_verifier import accept_greedy_batch, results_from_payload
from hybridinfer.spec_decode.commit import commit_batch
from hybridinfer.spec_decode.config import SpeculativeConfig
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.spec_decode.metadata import VerificationBatch
from hybridinfer.spec_decode.verifier import accept_greedy
from hybridinfer.utils.context import get_context


class BatchContractTests(unittest.TestCase):
    def test_hidden_and_logits_coordinate_systems_with_prefill_rows(self):
        plans = tuple(VerificationPlan(i, 3, 9, tuple(range(10, 10+k)))
                      for i, k in enumerate((3, 0, 2)))
        batch = VerificationBatch(plans, (4, 100, 3))
        m = batch.tensors('cpu')
        self.assertEqual(m.logits_indices.tolist(), [0, 1, 2, 3, 103, 104, 105, 106])
        self.assertEqual(m.target_logits_indices.tolist(), [0, 1, 2, 5, 6])
        self.assertEqual(m.bonus_logits_indices.tolist(), [3, 4, 7])
        self.assertEqual(m.cu_num_draft_tokens.tolist(), [3, 3, 5])
        self.assertEqual(m.cu_num_sampled_tokens.tolist(), [4, 5, 8])
        hidden = torch.arange(107)[:, None]
        selected = m.select_logits(hidden, lambda x: x)
        self.assertEqual(selected[m.bonus_logits_indices, 0].tolist(), [3, 103, 106])
        inputs = torch.arange(107)
        self.assertEqual(inputs[m.logits_indices][m.target_logits_indices+1].tolist(),
                         [1, 2, 3, 105, 106])

    def check_oracle(self, device):
        # Includes later matches after a rejection, K=0, every acceptance
        # endpoint, EOS in both candidates/correction, and output truncation.
        for counts in ((3, 0, 2), (0, 1, 4), (0, 0, 0)):
            plans = tuple(VerificationPlan(i, 3+i, 9, tuple(range(10, 10+k)))
                          for i, k in enumerate(counts))
            batch = VerificationBatch.from_plans(plans)
            m = batch.tensors(device)
            for accepted in product(*(range(k+1) for k in counts)):
                rows = []
                for p, a in zip(plans, accepted):
                    row = list(p.candidates)+( [99])
                    if a < len(p.candidates):
                        row[a] = 88
                    rows.append(row)
                for eos, remaining in ((-1, (10, 10, 10)), (11, (10, 10, 10)),
                                       (88, (10, 10, 10)), (99, (10, 10, 10)),
                                       (-1, (1, 2, 3))):
                    ignore = (False, True, False)
                    result = accept_greedy_batch(batch, m, torch.tensor(sum(rows, []), device=device),
                                                 remaining_output_tokens=remaining, max_model_len=10,
                                                 eos=eos, ignore_eos=ignore)
                    actual = results_from_payload(result.payload().cpu(), plans)
                    expected = [accept_greedy(p, row, remaining_output_tokens=r, max_model_len=10,
                                              eos=eos, ignore_eos=ig)
                                for p, row, r, ig in zip(plans, rows, remaining, ignore)]
                    self.assertEqual(actual, expected)

    def test_cpu_matches_independent_reference(self):
        self.check_oracle('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_matches_independent_reference(self):
        self.check_oracle('cuda')

    def test_invalid_layout_and_budgets(self):
        p = VerificationPlan(1, 3, 9, (1, 2))
        for plans, counts in (((), ()), ((p, p), (3, 3)), ((p,), (2,))):
            with self.assertRaises(ValueError):
                VerificationBatch(plans, counts)
        batch = VerificationBatch.from_plans([p])
        for predictions, remaining in (([1, 2], [5]), ([1, 2, 3], [0])):
            with self.assertRaises(ValueError):
                accept_greedy_batch(batch, batch.tensors('cpu'), torch.tensor(predictions),
                                    remaining_output_tokens=remaining, max_model_len=20)


def fake_batch(mode='packed', drift=False, fail_replay=False, device='cpu'):
    n = 4
    seqs = [Sequence([1, 2, 3, 10+i], SamplingParams(temperature=0, max_tokens=10)) for i in range(3)]
    slots = (2, 0, 3)  # Input order deliberately disagrees with resident slots.
    batch = InputBatch(n)
    dummy = Sequence([1])
    batch.update([seqs[1], dummy, seqs[0], seqs[2]])
    tokens = torch.full((n, 25), -1, dtype=torch.int64, device=device)
    tables = torch.full((n, 6), -1, dtype=torch.int32, device=device)
    for seq, slot in zip(seqs, slots):
        seq.num_cached_tokens = 3
        seq.block_table = list(range(slot*6, slot*6+6))
        tokens[slot, :4] = torch.tensor(seq.token_ids, device=device)
        tables[slot] = torch.tensor(seq.block_table, device=device)
    layer = SimpleNamespace(conv_states=torch.ones(2*n, 2, device=device),
                            recurrent_states=torch.ones(2*n, 2, device=device))
    computed = torch.tensor([3, 1, 3, 3], dtype=torch.int32, device=device)
    state = SimpleNamespace(tokens=SimpleNamespace(tensor=tokens),
                            computed=SimpleNamespace(tensor=computed),
                            block_tables=SimpleNamespace(tensor=tables), update=lambda *args: None)
    plans = tuple(VerificationPlan(seq.seq_id, 3, seq.last_token, drafts)
                  for seq, drafts in zip(seqs, ((20, 21, 22), (), (30, 31))))
    predictions = {slots[0]: [20, 88, 22, 99], slots[1]: [77], slots[2]: [30, 31, 99]}
    sources = {n+i: slot for i, slot in enumerate(slots)}
    kv = torch.zeros(2, 1, 32, 4, 1, 1, device=device)
    calls = []

    class Model:
        def __call__(self, ids, positions):
            ctx = get_context()
            calls.append(('packed' if ctx.is_prefill else 'decode', len(ids)))
            slices = ctx.prefill_slices if ctx.is_prefill else [(i, i+1) for i in range(len(ids))]
            targets = []
            flat = kv.reshape(2, 1, -1, 1, 1)
            for row, (begin, end) in enumerate(slices):
                dst = int(ctx.state_indices[row])
                src = sources.get(dst, dst)
                for index in range(begin, end):
                    token, position = int(ids[index]), int(positions[index])
                    for pool in (layer.conv_states, layer.recurrent_states):
                        pool[dst].add_(token)
                    if fail_replay and dst < n:
                        raise RuntimeError('replay failed')
                    flat[:, :, int(ctx.slot_mapping[index])] = layer.recurrent_states[dst, 0]
                    targets.append(predictions[src][position-3])
                if drift and ctx.is_prefill:
                    layer.recurrent_states[dst].add_(0.01)
            return torch.tensor(targets, device=device)[:, None]

        def compute_logits(self, hidden):
            logits = torch.full((len(hidden), 128), -100., device=device)
            logits.scatter_(1, hidden.to(torch.int64), 100.)
            return logits

    runner = SimpleNamespace(_pending=None, world_size=1, input_batch=batch, request_state=state,
                             config=SimpleNamespace(max_num_seqs=n, max_model_len=24, eos=-1,
                                                    speculative=SpeculativeConfig(enabled=True, verification_mode=mode)),
                             gdn_layers=[layer], block_size=4, model=Model(), calls=calls, kv_cache=kv,
                             sampled_token_ids_gpu=torch.zeros(n, dtype=torch.int64, device=device),
                             spec_metrics={}, async_output=True,
                             output_copy_stream=torch.cuda.Stream() if device=='cuda' else None)
    return runner, seqs, plans, slots


class BatchExecutionTests(unittest.TestCase):
    def check_batch(self, device, mode='packed', drift=False):
        runner, seqs, plans, slots = fake_batch(mode, drift=drift, device=device)
        before = [s.token_ids.copy() for s in seqs]
        handle = verify_speculative_batch(runner, seqs, plans)
        results = handle.get_output()
        self.assertIs(handle.get_output(), results)
        expected_tokens = [(20, 88), (77,), (30, 31, 99)] if mode == 'packed' else [(20,), (77,), (30,)]
        self.assertEqual([r.token_ids for r in results], expected_tokens)
        self.assertEqual([r.accepted_draft_tokens for r in results], [1, 0, 2] if mode == 'packed' else [0, 0, 0])
        self.assertEqual([r.committed_computed_length for r in results], [5, 4, 6] if mode == 'packed' else [4, 4, 4])
        self.assertEqual([r.trial_computed_tokens for r in results], [1, 1, 1] if mode == 'sequential' else [4, 1, 3])
        self.assertEqual([s.token_ids for s in seqs], before)
        for seq, plan, result, slot in zip(seqs, plans, results, slots):
            self.assertEqual(runner.request_state.tokens.tensor[slot, 4:4+result.output_length].tolist(),
                             list(result.token_ids))
            expected = 1+sum(plan.input_tokens[:result.output_length])
            actual = float(runner.gdn_layers[0].recurrent_states[slot, 0])
            self.assertAlmostEqual(actual, expected+(0.01 if drift and mode=='packed' else 0), places=3)
        self.assertEqual(runner.request_state.tokens.tensor[1].tolist(), [-1]*25)
        self.assertTrue(torch.equal(runner.gdn_layers[0].recurrent_states[1], torch.ones(2, device=device)))
        self.assertEqual(runner.spec_metrics['rounds'], 3)
        self.assertEqual(runner.spec_metrics['batch_zero_draft_requests'], 1)
        if mode == 'packed':
            self.assertEqual(runner.calls, [('packed', 8), ('packed', 2)])
            self.assertEqual(runner.spec_metrics['reference_trial_tokens'], 0)
        else:
            self.assertEqual(runner.calls, [('decode', 3)] if mode == 'sequential'
                             else [('packed', 8), ('decode', 3)])
            self.assertEqual(runner.spec_metrics['batch_reference_anchor_only'], 3)
        self.assertIsNone(get_context().state_indices)

    def test_native_single_forward_and_partial_replay(self):
        self.check_batch('cpu')

    def test_guarded_and_sequential_keep_reference_state(self):
        self.check_batch('cpu', 'packed_guarded', drift=True)
        self.check_batch('cpu', 'sequential')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_commit_and_async_output(self):
        self.check_batch('cuda')

    def test_replay_exception_rolls_back_all_requests(self):
        runner, seqs, plans, slots = fake_batch(fail_replay=True)
        tokens = runner.request_state.tokens.tensor.clone()
        computed = runner.request_state.computed.tensor.clone()
        with self.assertRaisesRegex(RuntimeError, 'replay failed'):
            verify_speculative_batch(runner, seqs, plans)
        self.assertTrue(torch.equal(runner.request_state.tokens.tensor, tokens))
        self.assertTrue(torch.equal(runner.request_state.computed.tensor, computed))
        for slot in slots:
            self.assertTrue(torch.equal(runner.gdn_layers[0].recurrent_states[slot], torch.ones(2)))

    def test_oom_restores_private_states_before_reference(self):
        runner, seqs, plans, slots = fake_batch()
        def oom(*args, **kwargs):
            runner.gdn_layers[0].recurrent_states[4:].add_(1000)
            raise torch.cuda.OutOfMemoryError('test OOM')
        with patch('hybridinfer.spec_decode.batch_execution.packed_batch_forward', side_effect=oom):
            results = verify_speculative_batch(runner, seqs, plans).get_output()
        self.assertEqual([r.token_ids for r in results], [(20,), (77,), (30,)])
        self.assertEqual(runner.spec_metrics['packed_resource_fallbacks'], 3)
        self.assertEqual(float(runner.gdn_layers[0].recurrent_states[slots[2], 0]), 13.)

    def test_output_handles_do_not_alias_later_commits(self):
        a, seqs_a, plans_a, _ = fake_batch()
        b, seqs_b, plans_b, _ = fake_batch()
        first = verify_speculative_batch(a, seqs_a, plans_a)
        second = verify_speculative_batch(b, seqs_b, plans_b)
        a.request_state.tokens.tensor.fill_(123)
        self.assertEqual(first.get_output(), second.get_output())


class BatchSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.old = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old

    def ready(self, blocks=32, budget=12):
        scheduler = Scheduler(SimpleNamespace(max_num_seqs=4, max_model_len=24,
                              max_num_batched_tokens=budget, eos=-1, kvcache_block_size=4,
                              num_kvcache_blocks=blocks, enable_prefix_cache=False,
                              speculative=SpeculativeConfig(enabled=True, ngram_max=2)))
        seqs = [Sequence(h, SamplingParams(temperature=0, max_tokens=10))
                for h in ([1, 2, 3, 4, 1, 2], [8, 9], [5, 6, 7, 5, 6])]
        for seq in seqs:
            scheduler.block_manager.allocate(seq, 0)
            seq.num_cached_tokens = seq.num_tokens-1
            seq.is_prefill = False
            scheduler.running.append(seq)
            scheduler.resident.add(seq.seq_id)
        return scheduler, seqs

    def test_batch_budget_in_flight_and_atomic_abort(self):
        scheduler, seqs = self.ready(budget=7)
        original = [list(s.block_table) for s in seqs]
        free = len(scheduler.block_manager.free_block_ids)
        selected, plans = scheduler.begin_speculative_batch()
        self.assertEqual(selected, seqs)
        self.assertEqual([len(p.candidates) for p in plans], [4, 0, 0])
        self.assertEqual(sum(len(p.input_tokens) for p in plans), 7)
        self.assertEqual(scheduler.in_flight, {s.seq_id for s in seqs})
        self.assertEqual(scheduler.schedule(), ([], False))
        scheduler.abort_speculative_batch(seqs)
        self.assertEqual([s.block_table for s in seqs], original)
        self.assertEqual(list(scheduler.running), seqs)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), free)
        self.assertFalse(scheduler.in_flight)

    def test_resource_failure_does_not_leave_partial_reservations(self):
        scheduler, seqs = self.ready()
        original = [list(s.block_table) for s in seqs]
        free = len(scheduler.block_manager.free_block_ids)
        reserve = scheduler.block_manager.reserve_trial
        calls = []
        def fail_second(seq, end):
            calls.append(seq.seq_id)
            return False if len(calls)==2 else reserve(seq, end)
        with patch.object(scheduler.block_manager, 'reserve_trial', side_effect=fail_second):
            self.assertIsNone(scheduler.begin_speculative_batch())
        self.assertEqual([s.block_table for s in seqs], original)
        self.assertEqual(len(scheduler.block_manager.free_block_ids), free)
        self.assertEqual(list(scheduler.running), seqs)
        self.assertFalse(scheduler.in_flight)

    def test_completed_batch_requeues_only_unfinished_requests(self):
        scheduler, seqs = self.ready()
        seqs[1].max_tokens = 1
        selected, plans = scheduler.begin_speculative_batch()
        for seq, plan in zip(selected, plans):
            result = accept_greedy(plan, list(plan.candidates)+[99],
                                   remaining_output_tokens=seq.max_tokens, max_model_len=24)
            scheduler.finish_speculative(seq, result)
            if not result.finished:
                self.assertEqual(seq.num_tokens, seq.num_cached_tokens+1)
        self.assertTrue(seqs[1].is_finished)
        self.assertEqual(list(scheduler.running), [seqs[0], seqs[2]])
        self.assertFalse(scheduler.in_flight)
        self.assertFalse(scheduler._spec_trial_block_counts)

    def test_in_flight_batch_allows_independent_prefill_and_later_preemption(self):
        scheduler, seqs = self.ready()
        selected, plans = scheduler.begin_speculative_batch()
        late = Sequence([4, 8, 9], SamplingParams(temperature=0, max_tokens=2))
        scheduler.add(late)
        scheduled, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(scheduled, [late])
        self.assertEqual(scheduler.in_flight, {s.seq_id for s in seqs+[late]})
        for seq, plan in zip(selected, plans):
            result = accept_greedy(plan, [88]*len(plan.input_tokens),
                                   remaining_output_tokens=10, max_model_len=24)
            scheduler.finish_speculative(seq, result)
        scheduler.postprocess(scheduled, [9], prefill)
        scheduler.running.remove(seqs[0])
        history = list(seqs[0].token_ids)
        scheduler.preempt(seqs[0])
        self.assertEqual(seqs[0].token_ids, history)
        self.assertEqual(seqs[0].num_cached_tokens, 0)
        self.assertEqual(scheduler.preempted, [seqs[0].seq_id])
        self.assertFalse(scheduler.in_flight)


class BatchEngineTests(unittest.TestCase):
    def test_engine_consumes_variable_outputs_after_completion_fence(self):
        from collections import deque
        from hybridinfer.engine.llm_engine import LLMEngine
        old = Sequence.block_size
        Sequence.block_size = 4
        try:
            runner, seqs, plans, slots = fake_batch()
            runner.config.max_num_batched_tokens = 12
            runner.config.num_kvcache_blocks = 32
            runner.config.kvcache_block_size = 4
            runner.config.enable_prefix_cache = False
            scheduler = Scheduler(runner.config)
            for seq in seqs:
                seq.block_table = []
                scheduler.block_manager.allocate(seq, 0)
                seq.num_cached_tokens = 3
                seq.is_prefill = False
                scheduler.running.append(seq)
                scheduler.resident.add(seq.seq_id)
            seqs[1].max_tokens = 1
            seqs[2].max_tokens = 3
            engine = LLMEngine.__new__(LLMEngine)
            engine.scheduler = scheduler
            engine.config = runner.config
            engine.model_runner = runner
            engine.batch_queue = deque()
            engine.max_concurrent_batches = 1
            removed = []
            completions = []
            def call(method, *args):
                if method == 'remove_request':
                    removed.append(args[0])
                    runner.input_batch.remove(args[0])
                    return
                self.assertEqual(method, 'verify_speculative_batch')
                output = verify_speculative_batch(runner, *args)
                get = output.get_output
                def consume():
                    self.assertEqual(scheduler.in_flight, {s.seq_id for s in seqs})
                    self.assertEqual([s.num_tokens for s in seqs], [4, 4, 4])
                    completions.append(True)
                    return get()
                output.get_output = consume
                return output
            runner.call = call
            drafts = {s.seq_id: p.candidates for s, p in zip(seqs, plans)}
            with patch('hybridinfer.spec_decode.ngram.NgramProposer._propose_one',
                       side_effect=lambda context: drafts[context.request_id]):
                outputs, count = engine.step()
            self.assertEqual(count, -6)
            self.assertEqual(completions, [True])
            self.assertEqual(removed, [seqs[1].seq_id, seqs[2].seq_id])
            self.assertEqual(outputs, [(seqs[1].seq_id, [77]), (seqs[2].seq_id, [30, 31, 99])])
            self.assertEqual(list(scheduler.running), [seqs[0]])
            self.assertFalse(scheduler.in_flight)
            self.assertFalse(engine.batch_queue)
        finally:
            Sequence.block_size = old
