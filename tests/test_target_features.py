import unittest
import torch
from torch import nn
from hybridinfer.models.qwen3_5 import Qwen3_5Model
from hybridinfer.utils.context import set_context, get_context, reset_context


class FeatureTests(unittest.TestCase):
    def test_checkpoint_boundary_order_residual_and_normalized_final(self):
        class Layer(nn.Module):
            def forward(self, positions, hidden_states, residual, prefill_slices):
                return hidden_states*2, hidden_states if residual is None else residual+3
        class Norm(nn.Module):
            def forward(self, hidden_states, residual):
                return (hidden_states+residual)/2, None
        model = Qwen3_5Model.__new__(Qwen3_5Model)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Embedding(4, 2)
        model.embed_tokens.weight.data.copy_(torch.arange(8).reshape(4, 2))
        model.layers = nn.ModuleList([Layer(), Layer(), Layer()])
        model.norm = Norm()
        model.has_linear_attention = False
        set_context(False)
        ctx = get_context()
        ctx.target_features = {}
        ctx.feature_layers = (2, 0)
        try:
            embedding = model.embed_tokens(torch.tensor([1, 2]))
            result = model(torch.tensor([1, 2]), torch.tensor([0, 1]))
            self.assertTrue(torch.equal(ctx.target_features[0], embedding))
            self.assertTrue(torch.equal(ctx.target_features[2], 5*embedding+3))
            self.assertTrue(torch.equal(result, (9*embedding+6)/2))
            self.assertTrue(torch.equal(ctx.target_features['final'], result))
            self.assertEqual(set(ctx.target_features), {0, 2, 'final'})
        finally:
            reset_context()
