"""Linear Llama EAGLE-3 drafter with private paged attention.

Architecture contract: SafeAILab/EAGLE eagle/model/cnets.py and vLLM
model_executor/models/llama_eagle3.py (Apache-2.0). This implementation uses
separate HF projection weights and the engine's existing attention kernels.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from hybridinfer.layers.attention import Attention


class EagleNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True)+self.eps)).to(x.dtype)*self.weight


class EagleAttention(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()
        self.num_heads, self.num_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
        self.head_dim = getattr(cfg, 'head_dim', cfg.hidden_size//self.num_heads)
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError('EAGLE-3 requires divisible GQA heads and even RoPE head_dim')
        bias = getattr(cfg, 'attention_bias', False)
        self.q_proj = nn.Linear(input_size, self.num_heads*self.head_dim, bias=bias)
        self.k_proj = nn.Linear(input_size, self.num_kv_heads*self.head_dim, bias=bias)
        self.v_proj = nn.Linear(input_size, self.num_kv_heads*self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, cfg.hidden_size, bias=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.head_dim**-.5, self.num_kv_heads)
        rope = getattr(cfg, 'rope_parameters', None) or {}
        self.rotary_dim = int(self.head_dim*rope.get('partial_rotary_factor', getattr(cfg, 'partial_rotary_factor', 1.)))
        if not 0 < self.rotary_dim <= self.head_dim or self.rotary_dim % 2:
            raise ValueError('Invalid EAGLE-3 rotary dimension')
        theta = rope.get('rope_theta', getattr(cfg, 'rope_theta', 10000.))
        inv = 1/theta**(torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)/self.rotary_dim)
        self.register_buffer('inv_freq', inv, persistent=False)

    def forward(self, positions, x):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        freq = positions.float()[:, None]*self.inv_freq[None]
        cos, sin = freq.cos()[:, None], freq.sin()[:, None]
        def rotate(value):
            a, b = value[..., :self.rotary_dim].float().chunk(2, -1)
            rotated = torch.cat((a*cos-b*sin, b*cos+a*sin), -1).to(value.dtype)
            return torch.cat((rotated, value[..., self.rotary_dim:]), -1)
        out = self.attn(rotate(q), rotate(k), v)
        return self.o_proj(out.reshape(x.shape[0], -1))


class EagleMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))


class EagleLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.index = index
        self.input_layernorm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.hidden_norm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = EagleAttention(cfg, cfg.hidden_size*(2 if index == 0 else 1))
        self.mlp = EagleMLP(cfg)
        self.norm_before_residual = getattr(cfg, 'norm_before_residual', False)

    def forward(self, embeddings, hidden, positions):
        if self.index == 0:
            normalized = self.hidden_norm(hidden)
            residual = normalized if self.norm_before_residual else hidden
            value = torch.cat((self.input_layernorm(embeddings), normalized), -1)
        else:
            residual, value = hidden, self.input_layernorm(hidden)
        hidden = residual+self.self_attn(positions, value)
        return hidden+self.mlp(self.post_attention_layernorm(hidden))


class Eagle3Draft(nn.Module):
    def __init__(self, config, target_hidden_size, target_vocab_size, feature_count):
        super().__init__()
        cfg = self.config = SimpleNamespace(**config)
        rope = config.get('rope_parameters') or {}
        if (getattr(cfg, 'hidden_act', 'silu') != 'silu' or getattr(cfg, 'rope_scaling', None)
                or rope.get('rope_type', 'default') != 'default'):
            raise ValueError('Initial EAGLE-3 supports SiLU and unscaled RoPE only')
        if getattr(cfg, 'quantization_config', None) or getattr(cfg, 'use_qk_norm', False):
            raise ValueError('Quantized/QK-normalized EAGLE-3 checkpoints are not supported')
        if getattr(cfg, 'mlp_bias', False) or getattr(cfg, 'sliding_window', None):
            raise ValueError('Biased MLP/sliding-window EAGLE-3 is not supported')
        if cfg.vocab_size != target_vocab_size or getattr(cfg, 'target_hidden_size', target_hidden_size) != target_hidden_size:
            raise ValueError('EAGLE-3 target vocabulary/hidden size mismatch')
        if cfg.num_hidden_layers < 1:
            raise ValueError('EAGLE-3 requires at least one draft decoder layer')
        eagle = config.get('eagle_config') or {}
        if not eagle.get('use_aux_hidden_state', True):
            raise ValueError('EAGLE-3 requires auxiliary target features')
        if (not eagle.get('use_input_layernorm_in_first_layer', True)
                or not eagle.get('use_last_layernorm', True) or eagle.get('use_mtp_layernorm', False)):
            raise ValueError('Unsupported EAGLE-3 normalization layout')
        if getattr(cfg, 'num_aux_hidden_states', feature_count) != feature_count:
            raise ValueError('EAGLE-3 feature count mismatch')
        self.feature_count = feature_count
        self.fc = nn.Linear(target_hidden_size*feature_count, cfg.hidden_size, bias=False)
        self.input_norm = (EagleNorm(target_hidden_size*feature_count, cfg.rms_norm_eps)
                           if eagle.get('norm_before_fc', config.get('norm_before_fc', False)) else None)
        self.fc_norm = (nn.ModuleList([EagleNorm(target_hidden_size, cfg.rms_norm_eps) for _ in range(feature_count)])
                        if config.get('fc_norm', False) else None)
        self.embed_tokens = nn.Embedding(target_vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([EagleLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm_output = config.get('norm_output', False)
        self.draft_vocab_size = config.get('draft_vocab_size') or target_vocab_size
        if not 0 < self.draft_vocab_size <= target_vocab_size:
            raise ValueError('Invalid EAGLE-3 draft vocabulary size')
        self.lm_head = nn.Linear(cfg.hidden_size, self.draft_vocab_size, bias=False)
        self.register_buffer('d2t', torch.zeros(self.draft_vocab_size, dtype=torch.int64))

    def combine_features(self, features):
        value = torch.cat(features, -1)
        if self.input_norm is not None:
            value = self.input_norm(value)
        if self.fc_norm is not None:
            value = torch.cat([norm(x) for norm, x in zip(self.fc_norm, value.chunk(self.feature_count, -1))], -1)
        return self.fc(value)

    def forward(self, embeddings, hidden, positions):
        for layer in self.layers:
            hidden = layer(embeddings, hidden, positions)
        return self.norm(hidden) if self.norm_output else hidden

    def compute_logits(self, hidden):
        logits = self.lm_head(hidden if self.norm_output else self.norm(hidden))*getattr(self.config, 'logit_scale', 1.)
        if self.draft_vocab_size == self.config.vocab_size:
            return logits
        output = logits.new_full((logits.shape[0], self.config.vocab_size), -float('inf'))
        output[:, torch.arange(self.draft_vocab_size, device=logits.device)+self.d2t] = logits
        return output

    def load_checkpoint(self, directory, target_embedding):
        params = dict(self.named_parameters())
        params['d2t'] = self.d2t
        loaded = set()
        for filename in sorted(Path(directory).glob('*.safetensors')):
            with safe_open(str(filename), framework='pt', device='cpu') as shard:
                for original in shard.keys():
                    name = original.removeprefix('model.').replace('midlayer.', 'layers.0.')
                    if name == 't2d':
                        continue  # Training-only reverse lookup; inference uses d2t.
                    if name == 'draft_id_to_target_id':
                        name = 'd2t'
                    if name not in params or name in loaded:
                        raise ValueError(f'Unexpected/duplicate EAGLE-3 weight: {original}')
                    value = shard.get_tensor(original)
                    if value.shape != params[name].shape:
                        raise ValueError(f'EAGLE-3 weight shape mismatch: {original}')
                    if name == 'd2t' and value.dtype not in (torch.int64, torch.int32):
                        raise ValueError('EAGLE-3 d2t mapping must contain integer offsets')
                    params[name].data.copy_(value)
                    loaded.add(name)
        required = set(params)-{'d2t', 'embed_tokens.weight'}
        if self.draft_vocab_size != self.config.vocab_size:
            required.add('d2t')
        if required-loaded:
            raise ValueError(f'Missing EAGLE-3 weights: {sorted(required-loaded)}')
        if 'embed_tokens.weight' not in loaded:
            if target_embedding.weight.shape != self.embed_tokens.weight.shape:
                raise ValueError('Missing EAGLE-3 embeddings cannot share incompatible target embeddings')
            self.embed_tokens = target_embedding
        mapping = torch.arange(self.draft_vocab_size, device=self.d2t.device)+self.d2t
        if ((mapping < 0).any() or (mapping >= self.config.vocab_size).any()
                or mapping.unique().numel() != mapping.numel()):
            raise ValueError('EAGLE-3 vocabulary mapping must be unique and within target vocabulary')
        if self.draft_vocab_size == self.config.vocab_size and self.d2t.count_nonzero():
            raise ValueError('Full-vocabulary EAGLE-3 requires identity mapping')
        return {'loaded_parameters': len(loaded), 'checkpoint_weights': sorted(loaded),
                'shared_embedding': 'embed_tokens.weight' not in loaded}


def read_eagle_config(directory):
    return json.loads((Path(directory)/'config.json').read_text())
