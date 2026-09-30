import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from hybridinfer.models.block_draft import DFlashDraft, DSparkDraft, BlockAttention
from hybridinfer.spec_decode.block import BlockProposer
from hybridinfer.spec_decode import SpeculativeConfig


def config(dspark=False):
    base = dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, rms_norm_eps=1e-6,
                vocab_size=40, head_dim=8, layer_types=['full_attention']*2)
    if dspark:
        return dict(transformer_layer_config=base, block_size=4, mask_token_id=39,
                    sample_from_anchor=True, draft_vocab_size=5, markov_rank=3,
                    enable_confidence_head=True, confidence_head_with_markov=True)
    return dict(base, dflash_config=dict(block_size=4, mask_token_id=39))


class BlockDraftTests(unittest.TestCase):
    def test_attention_matches_independent_dense_reference_and_masks(self):
        torch.manual_seed(7)
        for causal, window in ((False, None), (True, None), (True, 3), (False, 3)):
            cfg = SimpleNamespace(**config())
            attn = BlockAttention(cfg, causal, window)
            context, noise = torch.randn(4, 16), torch.randn(3, 16)
            positions = torch.arange(4, 7)
            prefix = attn.context_kv(context, torch.arange(4))
            actual = attn(positions, noise, *prefix)
            q = attn.rotate(attn.q_norm(attn.q_proj(noise).view(3, 2, 8)), positions).transpose(0, 1)
            k, v = attn.context_kv(torch.cat((context, noise)), torch.arange(7))
            k, v = k.repeat_interleave(2, 1).transpose(0, 1), v.repeat_interleave(2, 1).transpose(0, 1)
            scores = q@k.transpose(-1, -2)/8**.5
            for i in range(3):
                for j in range(7):
                    delta = 4+i-j
                    if ((causal and delta < 0) or (window and (delta >= window or (not causal and -delta >= window)))):
                        scores[:, i, j] = -float('inf')
            expected = attn.o_proj((scores.softmax(-1)@v).transpose(0, 1).reshape(3, 16))
            torch.testing.assert_close(actual, expected)
            changed = noise.clone(); changed[-1] += torch.randn(16)*10
            modified = attn(positions, changed, *prefix)
            if causal:
                torch.testing.assert_close(actual[0], modified[0])
            else:
                self.assertFalse(torch.allclose(actual[0], modified[0]))

    def test_strict_real_layout_loader_and_shared_target_weights(self):
        for dspark in (False, True):
            method = 'dspark' if dspark else 'dflash'
            model_type = DSparkDraft if dspark else DFlashDraft
            model = model_type(config(dspark), 16, 40, 2)
            weights = {k:v.clone() for k,v in model.state_dict().items() if k != 'd2t' or dspark}
            if dspark:
                weights['d2t'] = torch.arange(5)*2
                weights['t2d'] = torch.zeros(40, dtype=torch.int64)
            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp)/'model.safetensors')
                save_file(weights, path)
                embedding, head = torch.nn.Embedding(40,16), torch.nn.Linear(16,40,bias=False)
                model.load_checkpoint(tmp, embedding, head)
                self.assertIs(model.embed_tokens, embedding)
                if not dspark:
                    self.assertIs(model.lm_head, head)
                else:
                    self.assertEqual(model.confidence_head.proj.weight.dtype, torch.float32)
                save_file({k:v for k,v in weights.items() if k != 'hidden_norm.weight'}, path)
                with self.assertRaisesRegex(ValueError, 'Missing'):
                    model.load_checkpoint(tmp, embedding, head)

    def test_candidate_layout_and_markov_uses_mapped_previous_target_id(self):
        model = DFlashDraft(config(), 16, 40, 2)
        model.lm_head = torch.nn.Linear(16,40,bias=False)
        hidden = torch.randn(4,16)
        tokens, confidence = model.candidates(hidden, torch.tensor(2), 3)
        torch.testing.assert_close(tokens, model.lm_head(hidden[1:]).argmax(-1))
        self.assertIsNone(confidence)
        model = DSparkDraft(config(True), 16,40,2)
        model.d2t.copy_(torch.arange(5)*2)
        seen = []
        original = model.markov_head.markov_w1.forward
        def capture(ids):
            seen.append(ids.clone()); return original(ids)
        with patch.object(model.markov_head.markov_w1, 'forward', side_effect=capture):
            tokens, confidence = model.candidates(hidden, torch.tensor(2), 3)
        self.assertEqual(seen[0].item(), 2)
        torch.testing.assert_close(torch.cat(seen[1:]), tokens[:-1])
        self.assertEqual(tokens.shape, (3,))
        self.assertTrue(torch.all((confidence >= 0)&(confidence <= 1)))

    def test_only_committed_features_enter_private_cache(self):
        proposer = BlockProposer.__new__(BlockProposer)
        proposer.model = DFlashDraft(config(), 16,40,2)
        proposer.features = torch.randn(2,12,16)
        proposer.validated = [0,0]
        for layer in proposer.model.layers:
            layer.self_attn.attn.k_cache = torch.empty(2,12,1,8)
            layer.self_attn.attn.v_cache = torch.empty(2,12,1,8)
        first = [(k.clone(),v.clone()) for k,v in proposer._context(1,4)]
        proposer.features[1,4:] += 100
        repeated = proposer._context(1,4)
        for before, after in zip(first, repeated):
            torch.testing.assert_close(before[0], after[0]); torch.testing.assert_close(before[1], after[1])
        proposer.features[1,4:6] = torch.randn(2,16)
        advanced = proposer._context(1,6)
        for layer, (k,v) in zip(proposer.model.layers, advanced):
            expected = layer.self_attn.context_kv(proposer.features[1,:6],torch.arange(6))
            torch.testing.assert_close(k, expected[0]); torch.testing.assert_close(v,expected[1])
        self.assertEqual(proposer.validated, [0,6])
        with self.assertRaisesRegex(RuntimeError, 'ahead'):
            proposer._context(1,3)

    def test_block_width_budgets_reordering_and_slot_reuse(self):
        for anchor_layout in (False, True):
            proposer = BlockProposer.__new__(BlockProposer)
            proposer.model = DSparkDraft(config(True), 16, 40, 2)
            proposer.model.sample_from_anchor = anchor_layout
            proposer.features = torch.randn(2,12,16)
            proposer.validated, proposer.owners = [0,0], [None,None]
            proposer.last_hidden, proposer.last_confidences = {}, {}
            tokens = torch.arange(24).reshape(2,12)
            proposer.runner = SimpleNamespace(
                input_batch=SimpleNamespace(seq_id_to_slot={10:0,20:1}),
                request_state=SimpleNamespace(tokens=SimpleNamespace(tensor=tokens)),
                config=SimpleNamespace(speculative=SimpleNamespace(max_draft_tokens=8)))
            def ctx(request, end=4, remaining=9, budget=9):
                return SimpleNamespace(request_id=request, computed_length=end,
                    remaining_output_tokens=remaining,max_model_len=12,verification_budget=budget)
            widths, anchors = [], []
            def forward(ids, positions, kv):
                widths.append(ids.numel()); anchors.append(ids[0].item())
                return torch.ones(ids.numel(),16)
            def candidates(hidden, anchor, count):
                return torch.arange(count), None
            with patch.object(proposer, '_context', return_value=[]), patch.object(proposer.model, 'forward', side_effect=forward), patch.object(proposer.model, 'candidates', side_effect=candidates):
                result = proposer.propose_device([ctx(20,remaining=2),ctx(10,budget=3)])
                self.assertEqual(result.request_ids,(20,10))
                self.assertEqual(result.offsets,(0,1,3))
                self.assertEqual(widths,[1+(not anchor_layout),2+(not anchor_layout)])
                self.assertEqual(anchors,[16,4])
                result = proposer.propose_device([ctx(10,end=10)])
                self.assertEqual(result.offsets,(0,1))
                self.assertEqual(widths[-1],1+(not anchor_layout))
                self.assertEqual(proposer.propose_device([ctx(10,remaining=1)]).offsets,(0,0))
            proposer.release(0)
            self.assertIsNone(proposer.owners[0])
            self.assertEqual(proposer.validated[0],0)
            self.assertNotIn(0,proposer.last_confidences)

    def test_backend_configuration(self):
        for method in ('dflash','dspark'):
            with self.assertRaises(ValueError):
                SpeculativeConfig(enabled=True, method=method)
            self.assertEqual(SpeculativeConfig(enabled=True, method=method,draft_model='/tmp').method,method)


if __name__ == '__main__':
    unittest.main()
