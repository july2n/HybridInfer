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
    def test_probabilistic_drafts_realized_q_retry_and_request_reordering(self):
        from types import SimpleNamespace
        from hybridinfer.spec_decode.mtp import MTPProposer
        from hybridinfer.spec_decode.interfaces import DraftContext
        from hybridinfer.sampling.rejection_sampler import categorical, random_uniform
        proposer = MTPProposer.__new__(MTPProposer)
        logits = torch.tensor([[0., 1., 2.]])
        proposer.runner = SimpleNamespace(
            config=SimpleNamespace(speculative=SimpleNamespace(max_draft_tokens=3, mtp_draft_sampling='random')),
            input_batch=SimpleNamespace(seq_id_to_slot={7: 0, 8: 1}),
            request_state=SimpleNamespace(tokens=SimpleNamespace(tensor=torch.ones(2, 10, dtype=torch.int64))),
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones(3, 1)), compute_logits=lambda h: logits))
        proposer.features = torch.zeros(2, 10, 1)
        proposer.validated, proposer.owners, proposer.last_hidden = [0, 0], [None, None], {}
        proposer._forward = lambda slot, ids, features, *args: features+1
        contexts = [DraftContext(7, (1, 1, 1, 1), 3, 8, 10, 4, .7, 42),
                    DraftContext(8, (1, 1, 1), 2, 8, 10, 3, 0, 17)]
        first = proposer.propose_device(contexts)
        q = torch.softmax(logits.float()/.7, -1)
        torch.testing.assert_close(first.probabilities[:3], q.expand(3, -1))
        torch.testing.assert_close(first.probabilities[3:], torch.tensor([[0., 0., 1.], [0., 0., 1.]]))
        expected = [int(categorical(q[0], random_uniform(42, 4+i, 'draft', 'cpu'))) for i in range(3)]
        self.assertEqual(first.to_host().tokens_for(0), tuple(expected))
        second = proposer.propose_device(contexts)
        self.assertTrue(torch.equal(first.token_ids, second.token_ids))
        self.assertTrue(torch.equal(first.probabilities, second.probabilities))
        reordered = proposer.propose_device(contexts[::-1])
        self.assertEqual(reordered.to_host().tokens_for(1), first.to_host().tokens_for(0))
        torch.testing.assert_close(reordered.probabilities[2:], first.probabilities[:3])
        empty = proposer.propose_device([])
        self.assertEqual(empty.probabilities.shape, (0, 3))
        zero = proposer.propose_device([DraftContext(7, (1, 1, 1, 1), 3, 1, 10, 4, 1, 42)])
        self.assertEqual(zero.probabilities.shape, (0, 3))

    def test_random_draft_config_requires_native_mtp(self):
        from hybridinfer.spec_decode import SpeculativeConfig
        for kwargs in [dict(method='ngram', verification_mode='packed', mtp_draft_sampling='random'),
                       dict(method='mtp', mtp_draft_sampling='random'), dict(mtp_draft_sampling='unknown')]:
            with self.assertRaises(ValueError):
                SpeculativeConfig(**kwargs)
        SpeculativeConfig(method='mtp', verification_mode='packed', mtp_draft_sampling='random')

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
