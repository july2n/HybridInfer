import unittest

from hybridinfer.spec_decode.config import SpeculativeConfig
from hybridinfer.spec_decode.interfaces import DraftContext, DraftProposal, VerificationPlan
from hybridinfer.spec_decode.verifier import accept_greedy


class ContractTests(unittest.TestCase):
    def test_disabled_default_and_validation(self):
        self.assertFalse(SpeculativeConfig().enabled)
        for kwargs in ({'method': 'mtp'}, {'max_draft_tokens': 0},
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
