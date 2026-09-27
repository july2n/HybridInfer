"""Reference semantics and checkpoint corruption checks for Dense support."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn
from safetensors.torch import save_file

from hybridinfer.utils.loader import load_model


class ConfigTests(unittest.TestCase):
    def config(self, **kwargs):
        from hybridinfer.config import Config
        settings = dict(model_type="qwen3_5_text", num_hidden_layers=4,
                        max_position_embeddings=4096, layer_types=None,
                        full_attention_interval=4, rope_parameters={"rope_type": "default"})
        settings.update(kwargs)
        with tempfile.TemporaryDirectory() as directory, patch(
            "hybridinfer.config.config.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(text_config=SimpleNamespace(**settings)),
        ):
            return Config(directory)

    def test_legacy_interval_sets_hybrid_and_layer_types(self):
        config = self.config()
        self.assertTrue(config.is_hybrid)
        self.assertEqual(config.hf_config.layer_types, ["linear_attention"] * 3 + ["full_attention"])

    def test_invalid_layer_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "layer_types"):
            self.config(layer_types=["full_attention"])

    def test_moe_and_nondefault_rope_are_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, "Dense"):
            self.config(model_type="qwen3_5_moe_text")
        with self.assertRaisesRegex(ValueError, "default.*RoPE"):
            self.config(rope_parameters={"rope_type": "yarn"})


class LoaderTests(unittest.TestCase):
    def load(self, model, weights):
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, str(Path(directory) / "model.safetensors"))
            return load_model(model, directory)

    def test_missing_text_weight_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing text weights.*bias"):
            self.load(nn.Linear(2, 2), {"weight": torch.ones(2, 2)})

    def test_unknown_text_weight_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected text weight"):
            self.load(nn.Linear(2, 2, bias=False), {"typo.weight": torch.ones(2, 2)})

    def test_known_auxiliary_weights_ignored(self):
        model = nn.Linear(2, 2, bias=False)
        self.load(model, {"weight": torch.ones(2, 2), "model.visual.weight": torch.zeros(1),
                          "mtp.weight": torch.zeros(1)})
        self.assertTrue(torch.equal(model.weight, torch.ones(2, 2)))

    def test_shared_storage_head_is_loaded_through_embedding(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(2, 2)
        model.lm_head = nn.Linear(2, 2, bias=False)
        model.lm_head.weight.data = model.model.embed_tokens.weight.data
        self.load(model, {"model.language_model.embed_tokens.weight": torch.ones(2, 2)})
        self.assertTrue(torch.equal(model.lm_head.weight, torch.ones(2, 2)))

    def test_incomplete_packed_projection_rejected(self):
        model = nn.Module()
        model.gate_up_proj = nn.Linear(2, 4, bias=False)
        model.packed_modules_mapping = {"gate_proj": ("gate_up_proj", 0), "up_proj": ("gate_up_proj", 1)}
        def loader(param, weight, shard):
            param.data[2 * shard:2 * (shard + 1)].copy_(weight)
        model.gate_up_proj.weight.weight_loader = loader
        with self.assertRaisesRegex(ValueError, "Incomplete packed weight"):
            self.load(model, {"gate_proj.weight": torch.ones(2, 2)})

    def test_invalid_gdn_state_dtype_rejected(self):
        from hybridinfer.layers.gated_delta_net import GatedDeltaNet
        config = SimpleNamespace(hidden_size=32, linear_num_value_heads=4, linear_num_key_heads=4,
                                 linear_key_head_dim=128, linear_value_head_dim=128,
                                 linear_conv_kernel_dim=4, rms_norm_eps=1e-6, mamba_ssm_dtype="float16")
        with self.assertRaisesRegex(ValueError, "state dtype"):
            GatedDeltaNet(config, 0)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import transformers.models.qwen3_5.modeling_qwen3_5 as hf
        except ImportError:
            raise unittest.SkipTest("Transformers with Qwen3.5 required")
        cls.hf = hf

    def test_norm_and_residual_match_transformers_bf16(self):
        from hybridinfer.layers.layernorm import GemmaRMSNorm
        generator = torch.Generator().manual_seed(42)
        ref = self.hf.Qwen3_5RMSNorm(64).to(torch.bfloat16)
        ours = GemmaRMSNorm(64).to(torch.bfloat16)
        with torch.no_grad():
            ref.weight.copy_(torch.randn(64, generator=generator) * 0.2)
            ours.weight.copy_(ref.weight)
        x = torch.randn(17, 64, generator=generator).to(torch.bfloat16)
        residual = torch.randn(17, 64, generator=generator).to(torch.bfloat16)
        saved = x.clone()
        self.assertTrue(torch.equal(ours(x), ref(x)))
        output, added = ours(x, residual)
        self.assertTrue(torch.equal(added, x + residual))
        self.assertTrue(torch.equal(output, ref(x + residual)))
        self.assertTrue(torch.equal(x, saved))

    def test_gated_norm_preserves_checkpoint_weight_precision(self):
        from hybridinfer.layers.layernorm import RMSNormGated
        generator = torch.Generator().manual_seed(29)
        x = torch.randn(9, 128, generator=generator).to(torch.bfloat16)
        z = torch.randn(9, 128, generator=generator).to(torch.bfloat16)
        for dtype in (torch.float32, torch.bfloat16):
            ref = self.hf.Qwen3_5RMSNormGated(128).to(dtype)
            ours = RMSNormGated(128).to(dtype)
            with torch.no_grad():
                ref.weight.copy_(torch.randn(128, generator=generator) * 0.1 + 1)
                ours.weight.copy_(ref.weight)
            self.assertTrue(torch.equal(ours(x, z), ref(x, z)))

    def test_partial_rope_matches_transformers_bf16(self):
        from transformers import Qwen3_5TextConfig
        from hybridinfer.layers.rotary_embedding import InterleavedMRoPE
        config = Qwen3_5TextConfig(head_dim=256)
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000000.,
                                  "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]}
        ref = self.hf.Qwen3_5TextRotaryEmbedding(config)
        ours = InterleavedMRoPE(256, 0.25, [11, 11, 10], 8193, 10000000.)
        generator = torch.Generator().manual_seed(7)
        positions = torch.tensor([0, 1, 63, 64, 257, 4095, 8192])
        query = torch.randn(7, 4, 256, generator=generator).to(torch.bfloat16)
        key = torch.randn(7, 2, 256, generator=generator).to(torch.bfloat16)
        cos, sin = ref(query, positions[None])
        rq, rk = self.hf.apply_rotary_pos_emb(query[None], key[None], cos, sin, unsqueeze_dim=2)
        q, k = ours(positions, query, key)
        self.assertTrue(torch.equal(q, rq[0]))
        self.assertTrue(torch.equal(k, rk[0]))
        self.assertTrue(torch.equal(q[..., 64:], query[..., 64:]))

    def test_attention_projection_head_gate_matches_transformers(self):
        from transformers import Qwen3_5TextConfig
        from hybridinfer.models.qwen3_5 import Qwen3_5Attention
        config = Qwen3_5TextConfig(hidden_size=32, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
        config.rope_parameters = {"rope_type": "default", "rope_theta": 10000000.,
                                  "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 2]}
        config._attn_implementation = "eager"
        reference = self.hf.Qwen3_5Attention(config, 0).to(torch.bfloat16).eval()
        with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
            ours = Qwen3_5Attention(config, 0).to(torch.bfloat16).eval()
        ours.load_state_dict(reference.state_dict(), strict=True)
        hf = self.hf
        class EagerCore(nn.Module):
            def forward(self, q, k, v):
                mask = torch.full((7, 7), float("-inf"), dtype=q.dtype).triu(1)
                output, _ = hf.eager_attention_forward(
                    reference, q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                    v.transpose(0, 1)[None], mask, reference.scaling)
                return output[0]
        ours.attn = EagerCore()
        x = torch.randn(7, 32, generator=torch.Generator().manual_seed(8)).to(torch.bfloat16)
        positions = torch.arange(7)
        cos, sin = self.hf.Qwen3_5TextRotaryEmbedding(config)(x, positions[None])
        mask = torch.full((7, 7), float("-inf"), dtype=x.dtype).triu(1)
        expected, _ = reference(x[None], (cos, sin), mask)
        self.assertTrue(torch.equal(ours(positions, x), expected[0]))

    def test_packed_mlp_matches_transformers_bf16(self):
        from transformers import Qwen3_5TextConfig
        from hybridinfer.models.qwen3_5 import Qwen3_5MLP
        config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64)
        reference = self.hf.Qwen3_5MLP(config, 64).to(torch.bfloat16)
        with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
            ours = Qwen3_5MLP(config).to(torch.bfloat16)
        with torch.no_grad():
            ours.gate_up_proj.weight.copy_(torch.cat([reference.gate_proj.weight, reference.up_proj.weight]))
            ours.down_proj.weight.copy_(reference.down_proj.weight)
        x = torch.randn(17, 32, generator=torch.Generator().manual_seed(31)).to(torch.bfloat16)
        self.assertTrue(torch.equal(ours(x), reference(x)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GDNKernelTests(unittest.TestCase):
    def test_state_pool_obeys_configured_precision_and_reset(self):
        from hybridinfer.layers.gated_delta_net import GatedDeltaNet
        for name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
            config = SimpleNamespace(hidden_size=32, linear_num_value_heads=4, linear_num_key_heads=4,
                                     linear_key_head_dim=128, linear_value_head_dim=128,
                                     linear_conv_kernel_dim=4, rms_norm_eps=1e-6, mamba_ssm_dtype=name)
            layer = GatedDeltaNet(config, 0).to(device="cuda", dtype=torch.bfloat16)
            layer.allocate_state_pool(2)
            self.assertEqual(layer.recurrent_states.dtype, dtype)
            self.assertEqual(layer.conv_states.dtype, torch.bfloat16)
            layer.conv_states.fill_(1)
            layer.recurrent_states.fill_(1)
            layer.reset_state([1])
            self.assertTrue(torch.equal(layer.recurrent_states[0], torch.ones_like(layer.recurrent_states[0])))
            self.assertEqual(int(layer.recurrent_states[1].count_nonzero()), 0)
            self.assertEqual(int(layer.conv_states[1].count_nonzero()), 0)

    def oracle(self, q, k, v, g, beta, state):
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
        def normalize(x):
            return (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype)
        return torch_recurrent_gated_delta_rule(
            normalize(q), normalize(k), v, g, beta,
            state.transpose(-1, -2).float(), True, False)

    def assert_relative_error(self, actual, expected, limit):
        self.assertTrue(torch.isfinite(actual).all())
        delta = (actual.float() - expected.float()).square().mean().sqrt()
        scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
        self.assertLessEqual(float(delta / scale), limit)

    def test_varlen_chunk_matches_recurrence_and_preserves_idle_slot(self):
        from hybridinfer.layers.gated_delta_net import chunk_gated_delta_rule
        generator = torch.Generator(device="cuda").manual_seed(31)
        q, k, v = [torch.randn(1, 68, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)]
        g = -torch.rand(1, 68, 4, device="cuda", generator=generator)
        beta = torch.rand(1, 68, 4, device="cuda", generator=generator)
        pool = torch.randn(3, 4, 128, 128, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
        before = pool.clone()
        actual = chunk_gated_delta_rule(q, k, v, g, beta, pool,
                                       torch.tensor([2, 0], device="cuda"),
                                       torch.tensor([0, 3, 68], device="cuda", dtype=torch.int32))
        for start, end, slot in ((0, 3, 2), (3, 68, 0)):
            expected, final = self.oracle(q[:, start:end], k[:, start:end], v[:, start:end],
                                          g[:, start:end], beta[:, start:end], before[slot:slot + 1])
            self.assert_relative_error(actual[:, start:end], expected, 0.02)
            self.assert_relative_error(pool[slot], final[0].transpose(-1, -2), 0.02)
        self.assertTrue(torch.equal(pool[1], before[1]))

    def test_flashinfer_decode_matches_independent_recurrence(self):
        from hybridinfer.layers.gated_delta_net import decode_gated_delta_rule
        generator = torch.Generator(device="cuda").manual_seed(41)
        q, k, v = [torch.randn(2, 1, 4, 128, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)]
        a, b = [torch.randn(2, 1, 4, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(2)]
        a_log = torch.randn(4, device="cuda", generator=generator)
        bias = torch.randn(4, device="cuda", dtype=torch.bfloat16, generator=generator)
        state = torch.randn(2, 4, 128, 128, device="cuda", dtype=torch.bfloat16, generator=generator) * 0.1
        g = -a_log.exp() * torch.nn.functional.softplus(a.float() + bias.float())
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(state_dtype=dtype):
                typed_state = state.to(dtype)
                actual, final = decode_gated_delta_rule(q, k, v, a, b, a_log, bias, typed_state.clone())
                expected, expected_final = self.oracle(q, k, v, g, b.float().sigmoid(), typed_state)
                self.assert_relative_error(actual, expected, 0.02)
                self.assert_relative_error(final, expected_final.transpose(-1, -2), 0.02)
