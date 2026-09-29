"""Real graph replay regressions for shared layer buffers and bucket changes."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hybridinfer.engine.cuda_graph import CudaGraphManager
from hybridinfer.utils.context import set_context, reset_context


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)
        self.out = nn.Linear(8, 8, bias=False)

    def forward_piecewise_pre(self, hidden, residual):
        residual = hidden if residual is None else hidden + residual
        return (self.proj(residual), residual * .25), residual

    def forward_attention_core(self, projections, positions, slices):
        # Reads only real rows, like a varlen attention core.
        a, b = projections
        return a + b + positions[:, None].to(a.dtype) * .001

    def forward_output(self, hidden, residual):
        residual = hidden + residual
        return self.out(residual), residual


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(32, 8)
        self.model.layers = nn.ModuleList([Layer() for _ in range(3)])
        self.model.norm = lambda hidden, residual: (hidden + residual, None)

    def forward(self, ids, positions):
        hidden = self.model.embed_tokens(ids)
        residual = None
        for layer in self.model.layers:
            pre, residual = layer.forward_piecewise_pre(hidden, residual)
            hidden = layer.forward_attention_core(pre, positions, [(0, len(ids))])
            hidden, residual = layer.forward_output(hidden, residual)
        return hidden + residual


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class PiecewiseGraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        model = Model().cuda()
        self.runner = SimpleNamespace(model=model, use_prefill_cudagraph=True,
                                      config=SimpleNamespace(max_num_batched_tokens=256,
                                                             hf_config=SimpleNamespace(hidden_size=8)),
                                      compute_logits=lambda hidden, prefill: hidden)
        self.manager = CudaGraphManager(self.runner)
        self.manager.capture_prefill([256, 128, 256])

    def tearDown(self):
        torch.cuda.synchronize()
        reset_context()
        self.manager.clear()

    @torch.inference_mode()
    def check_length(self, length):
        ids = torch.arange(length, device='cuda') % 32
        positions = torch.arange(length, device='cuda') + 7
        set_context(True, prefill_slices=[(0, length)])
        expected = self.runner.model(ids, positions)
        actual = self.manager.run_prefill(ids, positions).clone()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_replay_across_shrinking_growing_and_fallback_buckets(self):
        # Alternate buckets, partial padding, and eager fallback on one manager.
        for length in [255, 17, 129, 1, 128, 256, 300, 3, 200]:
            self.check_length(length)

    @torch.inference_mode()
    def test_decode_can_use_inner_piecewise_without_prefill_slices(self):
        from hybridinfer.engine.cuda_graph import CUDAGraphMode, graph_mode
        ids = torch.arange(17, device='cuda') % 32
        positions = torch.arange(17, device='cuda') + 3
        set_context(False)
        expected = self.runner.model(ids, positions)
        with graph_mode(CUDAGraphMode.PIECEWISE):
            actual = self.manager.full_wrapper(ids, positions).clone()
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_feature_boundary_keeps_snapshot_of_reused_buffers(self):
        from hybridinfer.utils.context import get_context
        ids = torch.arange(17, device='cuda') % 32
        positions = torch.arange(17, device='cuda') + 3
        set_context(True, prefill_slices=[(0, 17)])
        context = get_context()
        context.feature_layers = (0, 1)
        context.target_features = {}
        hidden = self.runner.model.model.embed_tokens(ids)
        first = hidden.clone()
        layer = self.runner.model.model.layers[0]
        pre, residual = layer.forward_piecewise_pre(hidden, None)
        hidden = layer.forward_attention_core(pre, positions, [(0, 17)])
        hidden, residual = layer.forward_output(hidden, residual)
        second = hidden + residual
        result = self.manager.run_prefill(ids, positions).clone()
        torch.testing.assert_close(context.target_features[0], first)
        torch.testing.assert_close(context.target_features[1], second)
        torch.testing.assert_close(context.target_features['final'], result)

    def test_recapture_and_shared_output_lifetime(self):
        self.check_length(129)
        self.manager.capture_prefill([32, 64])
        for length in [63, 7, 32, 64, 65]:
            self.check_length(length)
        self.assertEqual(len(self.manager.piecewise_buffers['pre_out']), 1)
        self.assertEqual(self.manager.prefill_graph_sizes, [32, 64])


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CompiledNestedGraphTests(unittest.TestCase):
    @torch.inference_mode()
    def test_shared_compiled_plan_across_none_full_and_piecewise(self):
        from hybridinfer.engine.cuda_graph import CUDAGraphMode, graph_mode
        model = Model().cuda()
        runner = SimpleNamespace(
            model=model, use_prefill_cudagraph=True,
            config=SimpleNamespace(enable_piecewise_compile=True, max_num_batched_tokens=128,
                                   hf_config=SimpleNamespace(hidden_size=8)),
            compute_logits=lambda hidden, prefill: hidden)
        manager = CudaGraphManager(runner)
        try:
            manager.capture_prefill([64, 128])
            for length in [1, 8, 17, 129, 3, 72, 1]:
                ids = torch.arange(length, device='cuda') % 32
                positions = torch.arange(length, device='cuda') + 3
                set_context(True, prefill_slices=[(0, length)])
                expected = model(ids, positions)
                for mode in [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE]:
                    with graph_mode(mode):
                        # FULL has no captured bucket here and must run the same
                        # compiled plan with inner wrappers passing through.
                        actual = manager.full_wrapper(ids, positions).clone()
                    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        finally:
            torch.cuda.synchronize()
            manager.clear()
            reset_context()


if __name__ == '__main__':
    unittest.main()
