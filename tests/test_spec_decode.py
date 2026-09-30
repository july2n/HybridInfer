import unittest

from hybridinfer.spec_decode.config import SpeculativeConfig
from hybridinfer.spec_decode.interfaces import DraftContext, DraftProposal, VerificationPlan
from hybridinfer.sampling.greedy_reference import accept_greedy


class ContractTests(unittest.TestCase):
    def test_disabled_default_and_validation(self):
        self.assertFalse(SpeculativeConfig().enabled)
        self.assertEqual(SpeculativeConfig(enabled=True).verification_mode, 'packed_guarded')
        for kwargs in ({'method': 'eagle3'}, {'max_draft_tokens': 0},
                       {'ngram_min': 3, 'ngram_max': 2}, {'verification_mode': 'batch'}):
            with self.assertRaises(ValueError):
                SpeculativeConfig(**kwargs)

    def test_packed_variable_lengths(self):
        proposal = DraftProposal((7, 3, 9), (1, 2, 4), (0, 2, 2, 3))
        self.assertEqual([proposal.tokens_for(i) for i in range(3)], [(1, 2), (), (4,)])
        with self.assertRaises(ValueError):
            DraftProposal((1,), (2,), (0, 2))
        with self.assertRaises(ValueError):
            DraftContext(1, (1, 2), 0, 4, 20, 5)

    def test_every_acceptance_length_and_row_alignment(self):
        for k in range(5):
            plan = VerificationPlan(1, 9, 2, tuple(range(10, 10+k)))
            for a in range(k+1):
                predictions = list(plan.candidates) + [99]
                if a < k:
                    predictions[a] = 88
                result = accept_greedy(plan, predictions, remaining_output_tokens=10, max_model_len=30)
                self.assertEqual(result.accepted_draft_tokens, a)
                self.assertEqual(result.token_ids, plan.candidates[:a] + (predictions[a],))
                self.assertEqual(result.committed_computed_length, 9+a+1)
                self.assertEqual(result.trial_computed_tokens, k+1)
                self.assertFalse(result.finished)

    def test_termination_endpoints(self):
        plan = VerificationPlan(1, 3, 4, (5, 6, 7))
        for eos, remaining, expected, accepted, reason in (
            (5, 10, (5,), 1, 'eos'), (6, 10, (5, 6), 2, 'eos'),
            (8, 10, (5, 6, 7, 8), 3, 'eos'),
            (-1, 2, (5, 6), 2, 'max_tokens')):
            r = accept_greedy(plan, (5, 6, 7, 8), eos=eos,
                              remaining_output_tokens=remaining, max_model_len=30)
            self.assertEqual((r.token_ids, r.accepted_draft_tokens, r.finish_reason),
                             (expected, accepted, reason))
            self.assertEqual(r.committed_computed_length, 3+len(expected))
        r = accept_greedy(plan, (5, 6, 7, 8), eos=5, ignore_eos=True,
                          remaining_output_tokens=10, max_model_len=7)
        self.assertEqual(r.finish_reason, 'context')
        self.assertEqual(r.committed_computed_length, 7)

    def test_invalid_rows_and_budget(self):
        plan = VerificationPlan(1, 3, 4, (5,))
        for predictions, remaining, limit in (((5,), 10, 30), ((5, 6), 0, 30), ((5, 6), 10, 4)):
            with self.assertRaises(ValueError):
                accept_greedy(plan, predictions, remaining_output_tokens=remaining, max_model_len=limit)


if __name__ == '__main__':
    unittest.main()


class NgramTests(unittest.TestCase):
    def draft(self, tokens, remaining=20, limit=100, budget=20, **kwargs):
        from hybridinfer.spec_decode.ngram import NgramProposer
        context = DraftContext(7, tuple(tokens), len(tokens)-1, remaining, limit, budget)
        return NgramProposer(SpeculativeConfig(**kwargs)).propose([context]).tokens_for(0)

    def test_no_match_and_short_history(self):
        self.assertEqual(self.draft([1]), ())
        self.assertEqual(self.draft([1, 2, 3]), ())

    def test_longest_then_nearest(self):
        self.assertEqual(self.draft([1, 2, 3, 8, 2, 3, 9, 1, 2, 3]), (8, 2, 3, 9))
        self.assertEqual(self.draft([1, 2, 8, 1, 2, 9, 1, 2]), (9, 1, 2))

    def test_overlap_is_finite(self):
        self.assertEqual(self.draft([1, 1, 1, 1], ngram_min=2, ngram_max=2), (1,))
        self.assertEqual(self.draft([1, 2, 1, 2], ngram_max=2), (1, 2))

    def test_budget_clipping(self):
        history = [1, 2, 3, 4, 5, 1, 2]
        self.assertEqual(self.draft(history, remaining=2), (3,))
        self.assertEqual(self.draft(history, limit=8), (3,))
        self.assertEqual(self.draft(history, budget=2), (3,))
        self.assertEqual(self.draft(history, max_draft_tokens=2), (3, 4))
        self.assertEqual(self.draft(history, remaining=1), ())

    def test_variable_batch_and_no_history_mutation(self):
        from hybridinfer.spec_decode.ngram import NgramProposer
        contexts = [DraftContext(7, (1, 2, 1, 2), 3, 10, 20, 5),
                    DraftContext(9, (8, 9), 1, 10, 20, 5)]
        before = list(contexts)
        result = NgramProposer(SpeculativeConfig(ngram_max=2)).propose(contexts)
        self.assertEqual(result.offsets, (0, 2, 2))
        self.assertEqual(contexts, before)
