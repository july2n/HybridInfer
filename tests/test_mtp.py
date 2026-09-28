import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file
from transformers import Qwen3_5TextConfig
from hybridinfer.models.qwen3_5_mtp import Qwen3_5MTP


class MTPWeightTests(unittest.TestCase):
    def make_model(self):
        config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, vocab_size=32, max_position_embeddings=32,
            layer_types=['full_attention'])
        with patch('torch.distributed.get_world_size', return_value=1), patch('torch.distributed.get_rank', return_value=0):
            return Qwen3_5MTP(config)

    def checkpoint(self, model):
        weights = {}
        for name, parameter in model.named_parameters():
            value = torch.randn_like(parameter)
            if 'gate_up_proj' in name:
                a, b = value.chunk(2, 0)
                weights['mtp.'+name.replace('gate_up_proj', 'gate_proj')] = a.clone()
                weights['mtp.'+name.replace('gate_up_proj', 'up_proj')] = b.clone()
            else:
                weights['mtp.'+name] = value
        return weights

    def test_complete_checkpoint_missing_extra_and_incomplete_mlp(self):
        model = self.make_model()
        weights = self.checkpoint(model)
        self.assertEqual(len(weights), 15)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.safetensors'
            save_file(weights, str(path))
            report = model.load_checkpoint(tmp)
            self.assertEqual(len(report['checkpoint_weights']), 15)
            self.assertEqual(report['loaded_parameters'], 14)
            self.assertFalse(any('embed_tokens' in n or 'lm_head' in n for n, _ in model.named_parameters()))
            for bad in ['mtp.norm.weight', 'mtp.layers.0.mlp.up_proj.weight']:
                missing = {k: v for k, v in weights.items() if k != bad}
                save_file(missing, str(path))
                with self.assertRaises(ValueError):
                    model.load_checkpoint(tmp)
            save_file(dict(weights, **{'mtp.unknown.weight': torch.zeros(1)}), str(path))
            with self.assertRaises(ValueError):
                model.load_checkpoint(tmp)


class MTPPreparationTests(unittest.TestCase):
    def test_shifted_ids_keep_target_positions_and_refresh_confirmed_features(self):
        from types import SimpleNamespace
        from hybridinfer.spec_decode.mtp import MTPProposer
        from hybridinfer.spec_decode.interfaces import DraftContext
        proposer = MTPProposer.__new__(MTPProposer)
        tokens = torch.tensor([[9, 4, 5, 6, 2, 7, 0, 0, 0, 0]])
        proposer.runner = SimpleNamespace(config=SimpleNamespace(speculative=SimpleNamespace(max_draft_tokens=3)),
            input_batch=SimpleNamespace(seq_id_to_slot={7: 0}),
            request_state=SimpleNamespace(tokens=SimpleNamespace(tensor=tokens)),
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(1)),
                compute_logits=lambda hidden: torch.tensor([[0., 0., 10.]])))
        proposer.features = torch.arange(10).float().reshape(1, 10, 1)
        proposer.validated, proposer.owners, proposer.last_hidden = [0], [None], {}
        calls = []
        def forward(slot, ids, features, positions, physical, length, prefill):
            calls.append((ids.tolist(), features.clone(), positions.tolist(), physical.tolist(), length, prefill))
            return features+100
        proposer._forward = forward
        first = DraftContext(7, (9, 4, 5, 6), 3, 8, 10, 4)
        self.assertEqual(proposer.propose([first]).token_ids, (2, 2, 2))
        self.assertEqual(calls[0][0], [4, 5, 6])
        self.assertEqual(calls[0][2:4], ([0, 1, 2], [0, 1, 2]))
        self.assertEqual(calls[1][2:4], ([3], [3]))
        self.assertEqual(calls[2][2:4], ([4], [4]))
        self.assertEqual(proposer.last_hidden[0].item(), 102)
        # Abort/retry cannot expose autoregressive hidden or stale KV tail.
        calls.clear()
        proposer.propose([first])
        self.assertEqual(len(calls), 2)
        self.assertEqual(proposer.last_hidden[0].item(), 102)
        calls.clear()
        after = DraftContext(7, (9, 4, 5, 6, 2, 7), 5, 6, 10, 4)
        proposer.propose([after])
        self.assertEqual(calls[0][0], [2, 7])
        self.assertEqual(calls[0][2:4], ([3, 4], [3, 4]))
        self.assertEqual(calls[0][1].flatten().tolist(), [3., 4.])
        self.assertEqual(proposer.validated, [5])
        proposer.release(0)
        self.assertEqual(proposer.validated, [0])
        self.assertEqual(proposer.owners, [None])
        self.assertFalse(proposer.last_hidden)
