"""Diagnostics must report actual greedy winners, even at BF16 ties."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from validate_spec_natural import prediction_diagnostics
from spec_workloads import prepare_prompt
from bench_spec_decode import summarize_samples


class SpecDiagnosticTests(unittest.TestCase):
    def test_three_way_tie_reports_argmax_not_topk_tie_order(self):
        logits = torch.tensor([[5., 5., 5., 4.], [2., 1., 3., 0.]], dtype=torch.bfloat16)
        rows = prediction_diagnostics(logits, [512, 513], 'packed_trial')
        self.assertEqual([r['token'] for r in rows], logits.argmax(-1).tolist())
        self.assertEqual(rows[0]['margin'], 0.)
        self.assertNotEqual(rows[0]['second_token'], rows[0]['token'])
        self.assertEqual(rows[1]['margin'], 1.)
        self.assertEqual([r['position'] for r in rows], [512, 513])
        self.assertFalse(any(r['committed'] for r in rows))

    def test_position_mismatch_cannot_silently_drop_prediction_rows(self):
        with self.assertRaises(ValueError):
            prediction_diagnostics(torch.ones(2, 3), [4], 'ordinary')

    def test_natural_prompts_are_never_repeated_to_fill_length(self):
        class Tokenizer:
            def encode(self, text):
                return [1, 2, 3]
        self.assertEqual(prepare_prompt(Tokenizer(), 'short', 8, 'truncate'), [1, 2, 3])
        self.assertEqual(prepare_prompt(Tokenizer(), 'short', 2, 'truncate'), [1, 2])
        self.assertEqual(prepare_prompt(Tokenizer(), 'short', 8, 'repeat_and_truncate'),
                         [1, 2, 3, 1, 2, 3, 1, 2])

    def test_different_outputs_do_not_produce_an_accepted_speedup(self):
        def sample(tokens, seconds):
            return dict(tokens=tokens, decode_seconds=seconds, generation_seconds=seconds,
                        decode_tokens_per_second=len(tokens)/seconds,
                        metrics=dict(draft_tokens=2, accepted_tokens=1))
        for tokens, first in (([1, 9], 1), ([1], 1)):
            result = summarize_samples([sample(tokens, 1.)], [sample([1, 2], 2.)])
            self.assertFalse(result['speedup_comparable'])
            self.assertIsNone(result['decode_speedup'])
            self.assertIsNone(result['median_paired_decode_speedup'])
            self.assertEqual(result['first_token_mismatches'], [first])
            self.assertEqual(result['measured_decode_time_ratio'], 2.)

    def test_unpaired_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_samples([], [])


if __name__ == '__main__':
    unittest.main()
