import unittest
from types import SimpleNamespace
import torch
from hybridinfer.layers.gated_delta_net import GatedDeltaNet, decode_gated_delta_rule
from hybridinfer.layers.gdn_kernels import packed_causal_conv, indexed_gdn_decode
from hybridinfer.spec_decode.endpoints import select_endpoints
from hybridinfer.utils.context import set_context, get_context, reset_context, BatchDescriptor


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class OriginalEndpointTests(unittest.TestCase):
    @torch.inference_mode()
    def test_ragged_all_endpoints_against_flashinfer_and_conv_history(self):
        cfg = SimpleNamespace(hidden_size=32, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
            rms_norm_eps=1e-6, mamba_ssm_dtype='float32')
        layer = GatedDeltaNet(cfg, 2).to(device='cuda', dtype=torch.bfloat16)
        layer.A_log.data = layer.A_log.data.float()
        layer.norm.weight.data = layer.norm.weight.data.float()
        layer.allocate_state_pool(8)
        layer.conv_states.normal_()
        layer.recurrent_states.normal_(std=.1)
        original_conv, original_rec = layer.conv_states.clone(), layer.recurrent_states.clone()
        lengths, slots, slices = [1, 2, 4, 8], [6, 1, 4, 0], [(0, 1), (1, 3), (3, 7), (7, 15)]
        raw = torch.randn(15, layer.conv_dim, dtype=torch.bfloat16, device='cuda')
        z = torch.randn(15, layer.value_dim, dtype=torch.bfloat16, device='cuda')
        b = torch.randn(15, 4, dtype=torch.bfloat16, device='cuda')
        a = torch.randn_like(b)
        set_context(True, state_indices=torch.tensor(slots, device='cuda'), prefill_slices=slices,
            cu_seqlens_q=torch.tensor([0, 1, 3, 7, 15], device='cuda', dtype=torch.int32),
            batch_descriptor=BatchDescriptor('spec_decode', 15, 4, None, 8))
        get_context().state_endpoints = {}
        try:
            layer.forward_core_from_dense((raw, z, b, a))
            endpoints = get_context().state_endpoints
            _, conv, recurrent = endpoints[2]
            for slot, (start, end) in zip(slots, slices):
                reference_conv = original_conv.clone()
                state = original_rec[slot:slot+1].clone()
                indexed_pool = original_rec.clone()
                for t in range(start, end):
                    combined = torch.cat((reference_conv[slot], raw[t, :, None]), -1)
                    mixed = packed_causal_conv(raw[t:t+1], layer.conv1d.weight, reference_conv,
                        torch.tensor([slot], device='cuda'), None, 1, decode=True, round_before_silu=False)
                    q, k, v = torch.split(mixed, [layer.key_dim, layer.key_dim, layer.value_dim], -1)
                    indexed_gdn_decode(q.reshape(1, 1, 2, 128).contiguous(),
                        k.reshape(1, 1, 2, 128).contiguous(), v.reshape(1, 1, 4, 128).contiguous(),
                        a[t:t+1], b[t:t+1], layer.A_log, layer.dt_bias, indexed_pool,
                        torch.tensor([slot], device='cuda'))
                    # Packed uses vLLM fusion/warp layout; ordinary decode is
                    # the older non-fused path. The pinned vLLM differential
                    # benchmark separately requires exact equality.
                    torch.testing.assert_close(recurrent[t], indexed_pool[slot], rtol=1e-4, atol=1e-5)
                    q = q.reshape(1, 1, 2, 128).repeat_interleave(2, 2)
                    k = k.reshape(1, 1, 2, 128).repeat_interleave(2, 2)
                    _, state = decode_gated_delta_rule(q, k, v.reshape(1, 1, 4, 128),
                        a[t:t+1].clone(), b[t:t+1].clone(), layer.A_log, layer.dt_bias, state)
                    self.assertTrue(torch.equal(conv[t], combined[:, -3:]))
                    torch.testing.assert_close(recurrent[t], state[0], rtol=1e-4, atol=1e-5)
                    # Each possible accepted endpoint commits the original
                    # trial snapshot; untouched formal slots remain unchanged.
                    select_endpoints(endpoints, torch.tensor([7], device='cuda'),
                                     torch.tensor([t], device='cuda'))
                    self.assertTrue(torch.equal(layer.recurrent_states[7], recurrent[t]))
                    self.assertTrue(torch.equal(layer.conv_states[7], conv[t]))
            self.assertTrue(torch.equal(layer.recurrent_states[[2, 3, 5]], original_rec[[2, 3, 5]]))
        finally:
            reset_context()
