import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from hybridinfer.models.eagle3 import Eagle3Draft
from hybridinfer.spec_decode.eagle3 import Eagle3Proposer
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.utils.context import get_context, set_context, reset_context


def config():
    return dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-6,
                vocab_size=48, draft_vocab_size=8, rope_theta=10000.,
                target_hidden_size=16, eagle_config={'eagle_aux_hidden_state_layer_ids': [0, 1, 3]})


class Eagle3Tests(unittest.TestCase):
    def test_first_layer_concat_residual_and_feedback_norm_contract(self):
        from hybridinfer.models.eagle3 import EagleLayer
        cfg = config()
        embeddings, hidden = torch.randn(2, 32), torch.randn(2, 32)
        for before in (False, True):
            layer = EagleLayer(SimpleNamespace(**dict(cfg, norm_before_residual=before)), 0)
            capture = []
            def attention(positions, value):
                capture.append(value)
                return torch.zeros_like(hidden)
            def norm(x):
                return x*torch.rsqrt(x.square().mean(-1, keepdim=True)+cfg['rms_norm_eps'])
            with patch.object(layer.self_attn, 'forward', side_effect=attention), patch.object(layer.mlp, 'forward', return_value=torch.zeros_like(hidden)):
                output = layer(embeddings, hidden, torch.arange(2))
            torch.testing.assert_close(capture[0], torch.cat((norm(embeddings), norm(hidden)), -1))
            torch.testing.assert_close(output, norm(hidden) if before else hidden)
        for normalized in (False, True):
            model = Eagle3Draft(dict(cfg, norm_output=normalized), 16, 48, 3)
            with patch.object(model.layers[0], 'forward', return_value=hidden), patch.object(model.layers[1], 'forward', return_value=hidden):
                output = model(embeddings, hidden, torch.arange(2))
            torch.testing.assert_close(output, model.norm(hidden) if normalized else hidden)

    def test_strict_loading_embedding_and_offset_vocabulary_map(self):
        model = Eagle3Draft(config(), 16, 48, 3)
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        weights['d2t'] = torch.arange(8, dtype=torch.int64)*2
        weights['lm_head.weight'] = torch.arange(8*32).float().reshape(8, 32)/1000
        weights = {k.replace('layers.0.', 'midlayer.'): v for k, v in weights.items()}
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'model.safetensors'
            save_file(weights, str(file))
            model.load_checkpoint(tmp, torch.nn.Embedding(48, 16))
            hidden = torch.ones(1, 32)
            logits = model.compute_logits(hidden)
            targets = torch.arange(8)*3
            torch.testing.assert_close(logits[:, targets], model.lm_head(model.norm(hidden)))
            self.assertEqual(torch.isfinite(logits).sum(), 8)
            for bad in ['fc.weight', 'midlayer.self_attn.q_proj.weight', 'lm_head.weight', 'd2t']:
                save_file({k:v for k,v in weights.items() if k != bad}, str(file))
                with self.assertRaises(ValueError):
                    model.load_checkpoint(tmp, torch.nn.Embedding(48, 16))
            bad = dict(weights, d2t=-torch.arange(8))
            save_file(bad, str(file))
            with self.assertRaises(ValueError):
                model.load_checkpoint(tmp, torch.nn.Embedding(48, 16))
            no_embed = {k:v for k,v in weights.items() if k != 'embed_tokens.weight'}
            save_file(no_embed, str(file))
            with self.assertRaises(ValueError):
                model.load_checkpoint(tmp, torch.nn.Embedding(48, 16))
            embedding = torch.nn.Embedding(48, 32)
            report = model.load_checkpoint(tmp, embedding)
            self.assertTrue(report['shared_embedding'])
            self.assertIs(model.embed_tokens, embedding)

    def test_configuration_rejects_unsupported_architecture_and_missing_draft(self):
        with self.assertRaises(ValueError):
            SpeculativeConfig(enabled=True, method='eagle3')
        for override in [dict(rope_scaling={'type': 'linear', 'factor': 2}),
                         dict(partial_rotary_factor=.1), dict(target_hidden_size=15),
                         dict(num_aux_hidden_states=2), dict(hidden_act='gelu')]:
            with self.assertRaises(ValueError):
                Eagle3Draft(dict(config(), **override), 16, 48, 3)

    def test_feature_boundary_order_compression_and_context_restoration(self):
        proposer = Eagle3Proposer.__new__(Eagle3Proposer)
        proposer.pass_hidden_states_to_model = True
        proposer.feature_layers = (3, 0, 1)
        proposer.model = Eagle3Draft(config(), 16, 48, 3)
        features = [torch.ones(2, 16)*i for i in (3, 0, 1)]
        def target(ids, positions):
            context = get_context()
            for i, value in zip(proposer.feature_layers, features):
                context.target_features[i] = value
            return torch.ones(2, 16)
        proposer.runner = SimpleNamespace(model=target)
        proposer.features = torch.zeros(2, 8, 32)
        set_context(True)
        context = get_context()
        original = {'prior': torch.ones(1)}
        context.feature_layers, context.target_features = (2,), original
        try:
            hidden = proposer.forward_target(torch.tensor([1, 2]), torch.tensor([0, 1]))
            self.assertIs(context.target_features, original)
            self.assertEqual(context.feature_layers, (2,))
            expected = proposer.model.combine_features(features)
            proposer.record([1], torch.tensor([0, 1]), hidden, [(0, 2)])
            torch.testing.assert_close(proposer.features[1, :2], expected)
            self.assertIsNone(proposer._target_hidden)
            with patch.object(proposer.runner, 'model', side_effect=RuntimeError('target failed')):
                with self.assertRaises(RuntimeError):
                    proposer.forward_target(torch.tensor([1]), torch.tensor([0]))
            self.assertIs(context.target_features, original)
            self.assertIsNone(proposer._target_hidden)
        finally:
            reset_context()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_private_kv_chunked_and_decode_match_full_draft_forward(self):
        torch.manual_seed(42)
        model = Eagle3Draft(dict(config(), rope_parameters=dict(rope_type='default', rope_theta=10000000., partial_rotary_factor=.5)), 16, 48, 3).cuda().half()
        for layer in model.layers:
            attention = layer.self_attn.attn
            attention.k_cache = torch.zeros(1, 256, 1, 16, dtype=torch.float16, device='cuda')
            attention.v_cache = torch.zeros_like(attention.k_cache)
        ids = torch.arange(6, device='cuda')
        embeddings = model.embed_tokens(ids)
        features = model.combine_features([torch.randn(6, 16, device='cuda', dtype=torch.float16) for _ in range(3)])
        def run(start, end, decode=False):
            kwargs = dict(slot_mapping=torch.arange(start, end, device='cuda', dtype=torch.int32),
                          block_tables=torch.zeros(1, 1, device='cuda', dtype=torch.int32))
            if decode:
                kwargs['context_lens'] = torch.tensor([end], device='cuda', dtype=torch.int32)
            else:
                kwargs.update(cu_seqlens_q=torch.tensor([0, end-start], device='cuda', dtype=torch.int32),
                              cu_seqlens_k=torch.tensor([0, end], device='cuda', dtype=torch.int32),
                              max_seqlen_q=end-start, max_seqlen_k=end)
            set_context(not decode, **kwargs)
            return model(embeddings[start:end], features[start:end], ids[start:end])
        try:
            with torch.inference_mode():
                full = run(0, 6)
                chunked = torch.cat((run(0, 3), run(3, 5), run(5, 6, True)))
                torch.testing.assert_close(chunked, full, rtol=.005, atol=.005)
        finally:
            reset_context()
