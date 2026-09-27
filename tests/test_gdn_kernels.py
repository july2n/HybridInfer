"""Independent mathematical checks of packed convolution and indexed decode."""
import unittest
import torch
import torch.nn.functional as F

from hybridinfer.layers.gdn_kernels import packed_causal_conv, indexed_gdn_decode


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GDNKernelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_packed_conv_short_histories_boundaries_and_slot_reordering(self):
        for dtype in (torch.bfloat16, torch.float16):
            for lengths in ([1, 2, 3, 5, 17], [63, 64, 65], [129, 1]):
                torch.manual_seed(15)
                channels, kernel = 137, 4
                x = torch.randn(sum(lengths), channels, device='cuda', dtype=dtype)
                weights = torch.randn(channels, 1, kernel, device='cuda', dtype=dtype)
                pool = torch.randn(8, channels, kernel-1, device='cuda', dtype=dtype)
                old = pool.clone()
                slots = torch.tensor([6, 1, 4, 0, 3][:len(lengths)], device='cuda')
                bounds = [0]
                for length in lengths:
                    bounds.append(bounds[-1] + length)
                cu = torch.tensor(bounds, device='cuda', dtype=torch.int32)
                expected = []
                for req, slot in enumerate(slots.tolist()):
                    raw = x[bounds[req]:bounds[req+1]].T
                    combined = torch.cat((old[slot], raw), dim=1)
                    expected.append(F.silu(F.conv1d(combined[None], weights, groups=channels))[0].T)
                actual = packed_causal_conv(x, weights, pool, slots, cu, max(lengths))
                torch.testing.assert_close(actual, torch.cat(expected), rtol=.02, atol=.02)
                for req, slot in enumerate(slots.tolist()):
                    combined = torch.cat((old[slot], x[bounds[req]:bounds[req+1]].T), dim=1)
                    self.assertTrue(torch.equal(pool[slot], combined[:, -(kernel-1):]))
                idle = [i for i in range(8) if i not in slots.tolist()]
                self.assertTrue(torch.equal(old[idle], pool[idle]))

    @torch.inference_mode()
    def test_decode_conv_multiple_steps_and_cuda_graph(self):
        x = torch.randn(3, 129, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(129, 1, 4, device='cuda', dtype=torch.bfloat16)
        pool = torch.randn(6, 129, 3, device='cuda', dtype=torch.bfloat16)
        slots = torch.tensor([4, 0, 2], device='cuda')
        for _ in range(2):
            packed_causal_conv(x, w, pool, slots, None, 1, decode=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = packed_causal_conv(x, w, pool, slots, None, 1, decode=True)
        for _ in range(3):
            old = pool.clone()
            combined = torch.cat((old.index_select(0, slots), x[:, :, None]), dim=-1)
            expected = F.silu(F.conv1d(combined, w, groups=129)).squeeze(-1)
            graph.replay()
            torch.testing.assert_close(result, expected, rtol=.02, atol=.02)
            self.assertTrue(torch.equal(pool.index_select(0, slots), combined[:, :, -3:]))

    @torch.inference_mode()
    def test_indexed_fp32_decode_against_formula_and_flashinfer(self):
        from hybridinfer.layers.gated_delta_net import decode_gated_delta_rule
        for batch, hq, hv in [(1, 2, 2), (3, 2, 4), (8, 4, 4)]:
            torch.manual_seed(28)
            q = torch.randn(batch, 1, hq, 128, device='cuda', dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn(batch, 1, hv, 128, device='cuda', dtype=torch.bfloat16)
            a = torch.randn(batch, 1, hv, device='cuda', dtype=torch.bfloat16)
            b = torch.randn_like(a)
            log = torch.randn(hv, device='cuda')
            bias = torch.randn(hv, device='cuda', dtype=torch.bfloat16)
            pool = torch.randn(batch + 2, hv, 128, 128, device='cuda') * .1
            slots = torch.arange(batch, device='cuda').flip(0) + 1
            for _ in range(3):
                old = pool.clone()
                initial = old.index_select(0, slots)
                qn = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
                kn = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
                qn = qn[:, 0].repeat_interleave(hv // hq, dim=1) / 128**.5
                kn = kn[:, 0].repeat_interleave(hv // hq, dim=1)
                decay = torch.exp(-log.exp() * F.softplus(a[:, 0].float() + bias.float()))
                h = initial * decay[:, :, None, None]
                delta = (v[:, 0].float() - (h * kn[:, :, None]).sum(-1)) * b[:, 0].float().sigmoid()[:, :, None]
                expected_state = h + delta[:, :, :, None] * kn[:, :, None]
                expected_out = (expected_state * qn[:, :, None]).sum(-1)[:, None].to(q.dtype)
                actual = indexed_gdn_decode(q, k, v, a, b, log, bias, pool, slots)
                torch.testing.assert_close(actual, expected_out, rtol=.02, atol=.002)
                torch.testing.assert_close(pool.index_select(0, slots), expected_state, rtol=1e-4, atol=1e-5)
                reference, reference_state = decode_gated_delta_rule(q, k, v, a, b, log, bias, initial)
                torch.testing.assert_close(actual, reference, rtol=.02, atol=.002)
                torch.testing.assert_close(pool.index_select(0, slots), reference_state, rtol=1e-4, atol=1e-5)
                self.assertTrue(torch.equal(pool[[0, batch+1]], old[[0, batch+1]]))

    @torch.inference_mode()
    def test_gqa_layer_pool_graph_matches_flashinfer_without_repeating_qk(self):
        import copy
        from types import SimpleNamespace
        from hybridinfer.layers.gated_delta_net import GatedDeltaNet
        from hybridinfer.utils.context import set_context, reset_context
        cfg = SimpleNamespace(hidden_size=32, linear_num_value_heads=4,
                              linear_num_key_heads=2, linear_key_head_dim=128,
                              linear_value_head_dim=128, linear_conv_kernel_dim=4,
                              rms_norm_eps=1e-6, mamba_ssm_dtype='float32')
        layer = GatedDeltaNet(cfg, 0).to(device='cuda', dtype=torch.bfloat16)
        layer.A_log.data = layer.A_log.data.float()
        layer.norm.weight.data = layer.norm.weight.data.float()
        layer.decode_backend = 'pool'
        layer.allocate_state_pool(5)
        slots = torch.tensor([4, 0, 2], device='cuda')
        hidden = torch.randn(3, 1, 32, device='cuda', dtype=torch.bfloat16)
        set_context(False, state_indices=slots)
        try:
            for _ in range(3):
                layer(hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = layer(hidden)
            reference = copy.deepcopy(layer)
            reference.decode_backend = 'flashinfer'
            for _ in range(4):
                expected = reference(hidden)
                graph.replay()
                torch.testing.assert_close(output, expected, rtol=.03, atol=.003)
                torch.testing.assert_close(layer.recurrent_states, reference.recurrent_states,
                                           rtol=1e-4, atol=1e-5)
                self.assertTrue(torch.equal(layer.conv_states, reference.conv_states))
                self.assertEqual(layer.recurrent_states[[1, 3]].count_nonzero().item(), 0)
        finally:
            torch.cuda.synchronize()
            reset_context()
