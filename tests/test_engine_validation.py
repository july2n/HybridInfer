"""Regression checks for cross-mode validation and late-request diagnostics."""
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from validate_engine_qwen35 import _signature_comparison, signature_diagnostics, report_matrix
from hybridinfer.engine.cuda_graph import CudaGraphManager


def diag(tokens, logits):
    return {"top_tokens": tokens, "top_logits": logits,
            "effective_tolerance": 0.1, "logits_dtype": "bfloat16"}


class SignatureTests(unittest.TestCase):
    def test_rejected_matrix_writes_evidence_and_exits_nonzero(self):
        modes = ["BASELINE", "A", "B", "C"]
        per_mode = {name: {"compare_cases": {"probe": {"pass": True, "signature": [[1]]}}}
                    for name in modes}
        per_mode["B"]["compare_cases"]["probe"]["signature"] = [[2]]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = str(Path(directory) / "aggregate.json")
            with self.assertRaises(SystemExit) as error:
                report_matrix(per_mode, mode_names=modes, selected_compare=["probe"],
                              selected_local=[], base=path, meta={"gpu": "test"})
            self.assertEqual(error.exception.code, 1)
            result = json.loads(Path(path).read_text())
            self.assertFalse(result["overall_pass"])
            self.assertEqual(result["matrix"]["probe"]["B"]["status"], "MISMATCH")

    def test_capture_runs_without_autograd_and_restores_caller_mode(self):
        import torch
        manager = CudaGraphManager(SimpleNamespace(use_prefill_cudagraph=True))
        observed = []
        def check():
            observed.append(torch.is_inference_mode_enabled())
            self.assertFalse(torch.is_grad_enabled())
        with torch.enable_grad(), patch.object(manager, "_capture_decode", side_effect=check), patch.object(manager, "capture_prefill", side_effect=check):
            manager.capture()
            self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(torch.is_inference_mode_enabled())
        self.assertEqual(observed, [True, True])

    def test_prefill_capture_preserves_production_operator_callables(self):
        class Layer:
            def forward_piecewise_pre(self, hidden, residual):
                return hidden, residual
            def forward_output(self, hidden, residual):
                return hidden, residual

        layer = Layer()
        manager = CudaGraphManager(SimpleNamespace(config=SimpleNamespace()))
        with patch("torch.compile", side_effect=AssertionError("changes bf16 rounding")):
            for kind, original in (("pre", layer.forward_piecewise_pre),
                                   ("post", layer.forward_output)):
                captured = manager._piecewise_callable(layer, kind)
                self.assertEqual(captured, original)
                self.assertIs(captured, manager._piecewise_callable(layer, kind))

    def test_tie_does_not_hide_another_requests_clear_divergence(self):
        lhs = {"signature": [[1, 8], [3]], "signature_diagnostics": [
            [diag([1, 2], [5.0, 4.95]), None], [diag([3, 4], [6.0, 4.0])]]}
        rhs = {"signature": [[2, 9], [4]], "signature_diagnostics": [
            [diag([2, 1], [5.0, 4.95]), None], [diag([4, 3], [6.0, 4.0])]]}
        result = _signature_comparison(lhs, rhs)
        self.assertFalse(result["numeric_tie"])
        self.assertEqual(result["first_non_numeric_mismatch"]["path"], [1, 0])

    def test_post_divergence_tokens_do_not_reject_a_first_token_tie(self):
        lhs = {"signature": [[1, 8]], "signature_diagnostics": [
            [diag([1, 2], [5.0, 4.95]), None]]}
        rhs = {"signature": [[2, 9]], "signature_diagnostics": [
            [diag([2, 1], [5.0, 4.95]), None]]}
        self.assertTrue(_signature_comparison(lhs, rhs)["numeric_tie"])

    def test_length_difference_is_not_a_numeric_tie(self):
        result = _signature_comparison({"signature": [[1]]}, {"signature": [[1, 2]]})
        self.assertFalse(result["numeric_tie"])

    def test_three_way_tie_compares_actual_winners(self):
        lhs = {"signature": [[1]], "signature_diagnostics": [[diag([1, 2, 3], [5.0, 4.95, 4.95])]]}
        rhs = {"signature": [[3]], "signature_diagnostics": [[diag([2, 3, 1], [5.0, 5.0, 4.95])]]}
        self.assertTrue(_signature_comparison(lhs, rhs)["numeric_tie"])

    def test_close_unrelated_runner_up_cannot_hide_clear_winner_gap(self):
        lhs = {"signature": [[1]], "signature_diagnostics": [[diag([1, 2, 3], [5.0, 4.95, 3.0])]]}
        rhs = {"signature": [[3]], "signature_diagnostics": [[diag([3, 2, 1], [5.0, 4.95, 3.0])]]}
        self.assertFalse(_signature_comparison(lhs, rhs)["numeric_tie"])

    def test_request_count_difference_is_not_hidden_by_a_tie(self):
        lhs = {"signature": [[1]], "signature_diagnostics": [[diag([1, 2], [5, 5])]]}
        rhs = {"signature": [[2], [3]], "signature_diagnostics": [[diag([1, 2], [5, 5])]]}
        result = _signature_comparison(lhs, rhs)
        self.assertFalse(result["numeric_tie"])
        self.assertEqual(result["first_non_numeric_mismatch"]["length_mismatch"], [1, 2])

    def test_late_requests_are_bound_and_patch_restored_on_error(self):
        class Sampler:
            def enable_diagnostics(self, **kwargs):
                self.enabled = True
            def disable_diagnostics(self):
                self.enabled = False

        class Runner:
            sampler = Sampler()
            def execute_model(self, seqs, is_prefill):
                return [seq._diagnostic_key for seq in seqs]

        runner = Runner()
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with signature_diagnostics(SimpleNamespace(model_runner=runner)) as requests:
                runner.execute_model([SimpleNamespace(seq_id=10)], True)
                runner.execute_model([SimpleNamespace(seq_id=11)], True)
                self.assertEqual(set(requests), {10, 11})
                raise RuntimeError("probe")
        self.assertNotIn("execute_model", vars(runner))
        self.assertFalse(runner.sampler.enabled)
