"""DFlash/DSpark block drafters with feature-only context KV.

Architecture follows z-lab/dflash and vLLM Qwen3 DFlash/DSpark models.
"""
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

from .eagle3 import EagleAttention, EagleMLP, EagleNorm


class BlockAttention(EagleAttention):
    def __init__(self, cfg, causal, window):
        super().__init__(cfg, cfg.hidden_size)
        self.q_norm = EagleNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = EagleNorm(self.head_dim, cfg.rms_norm_eps)
        self.causal, self.window = causal, window

    def rotate(self, value, positions):
        freq = positions.float()[:, None]*self.inv_freq[None]
        cos, sin = freq.cos().to(value.dtype)[:, None], freq.sin().to(value.dtype)[:, None]
        a, b = value[..., :self.rotary_dim].chunk(2, -1)
        rotated = torch.cat((a*cos-b*sin, b*cos+a*sin), -1).to(value.dtype)
        return torch.cat((rotated, value[..., self.rotary_dim:]), -1)

    def context_kv(self, features, positions):
        k = self.k_norm(self.k_proj(features).view(-1, self.num_kv_heads, self.head_dim))
        v = self.v_proj(features).view(-1, self.num_kv_heads, self.head_dim)
        return self.rotate(k, positions), v

    def forward(self, positions, x, prefix_k, prefix_v):
        q = self.rotate(self.q_norm(self.q_proj(x).view(-1, self.num_heads, self.head_dim)), positions)
        k, v = self.context_kv(x, positions)
        k, v = torch.cat((prefix_k, k)), torch.cat((prefix_v, v))
        key_positions = torch.arange(k.shape[0], device=x.device)
        delta = positions[:, None]-key_positions[None, :]
        mask = torch.ones_like(delta, dtype=torch.bool)
        if self.causal:
            mask &= delta >= 0
        if self.window:
            mask &= delta < self.window
            if not self.causal:
                mask &= -delta < self.window
        repeat = self.num_heads//self.num_kv_heads
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
            k.repeat_interleave(repeat, 1).transpose(0, 1)[None],
            v.repeat_interleave(repeat, 1).transpose(0, 1)[None], attn_mask=mask)
        return self.o_proj(out[0].transpose(0, 1).reshape(x.shape[0], -1))


class BlockLayer(nn.Module):
    def __init__(self, cfg, kind, noncausal_sliding):
        super().__init__()
        if kind not in ('full_attention', 'sliding_attention'):
            raise ValueError(f'Unsupported block attention type: {kind}')
        sliding = kind == 'sliding_attention'
        self.input_layernorm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = BlockAttention(cfg, getattr(cfg, "is_causal", None) if getattr(cfg, "is_causal", None) is not None else sliding and not noncausal_sliding,
                                       getattr(cfg, 'sliding_window', None) if sliding else None)
        self.mlp = EagleMLP(cfg)

    def forward(self, hidden, positions, kv):
        hidden = hidden+self.self_attn(positions, self.input_layernorm(hidden), *kv)
        return hidden+self.mlp(self.post_attention_layernorm(hidden))


class BlockDraftBase(nn.Module):
    """Transformer and checkpoint handling shared by block draft models."""

    def __init__(self, raw, target_hidden, target_vocab, feature_count):
        super().__init__()
        cfg = self.config = SimpleNamespace(**raw.get('transformer_layer_config', raw))
        if (cfg.hidden_size != target_hidden or cfg.vocab_size != target_vocab
                or getattr(cfg, 'hidden_act', 'silu') != 'silu'
                or getattr(cfg, 'rope_scaling', None)
                or (getattr(cfg, 'rope_parameters', None) or {}).get('rope_type', 'default') != 'default'
                or getattr(cfg, 'quantization_config', None) or getattr(cfg, 'mlp_bias', False)):
            raise ValueError('Unsupported block draft dimensions, activation, RoPE or quantization')
        settings = raw.get('dflash_config', raw)
        self.block_size = settings.get('block_size', raw.get('block_size', 16))
        self.mask_token_id = settings['mask_token_id']
        if self.block_size < 2 or not 0 <= self.mask_token_id < target_vocab:
            raise ValueError('Invalid block size or mask token')
        self.fc = nn.Linear(target_hidden*feature_count, cfg.hidden_size, bias=False)
        self.hidden_norm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        kinds = getattr(cfg, 'layer_types', ['full_attention']*cfg.num_hidden_layers)
        if len(kinds) != cfg.num_hidden_layers:
            raise ValueError('Draft layer_types length mismatch')
        self.layers = nn.ModuleList([BlockLayer(cfg, k, raw.get('sliding_window_non_causal', False)) for k in kinds])
        self.norm = EagleNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.embed_tokens = None
        self.lm_head = None

    def combine_features(self, features):
        return self.hidden_norm(self.fc(torch.cat(features, -1)))

    def forward(self, ids, positions, prefix_kvs):
        hidden = self.embed_tokens(ids)
        for layer, kv in zip(self.layers, prefix_kvs):
            hidden = layer(hidden, positions, kv)
        return self.norm(hidden)

    def load_checkpoint(self, directory, target_embedding, target_head):
        params = dict(self.named_parameters())
        params.update(self.checkpoint_buffers())
        loaded = set()
        for filename in sorted(Path(directory).glob('*.safetensors')):
            with safe_open(str(filename), framework='pt', device='cpu') as shard:
                for original in shard.keys():
                    name = original.removeprefix('model.')
                    if name == 't2d':
                        continue
                    if name not in params or name in loaded:
                        raise ValueError(f'Unexpected/duplicate block draft weight: {original}')
                    value = shard.get_tensor(original)
                    if value.shape != params[name].shape:
                        raise ValueError(f'Block draft weight shape mismatch: {original}')
                    self.validate_weight(name, value)
                    params[name].data.copy_(value)
                    loaded.add(name)
        if set(params)-loaded:
            raise ValueError(f'Missing block draft weights: {sorted(set(params)-loaded)}')
        self.validate_checkpoint()
        expected = (self.config.vocab_size, self.config.hidden_size)
        if target_embedding.weight.shape != expected or target_head.weight.shape != expected:
            raise ValueError("Block draft cannot share incompatible target embeddings/head")
        self.embed_tokens = target_embedding
        self.attach_head(target_head)
        return {'loaded_parameters': len(loaded), 'shared_embedding': True,
                'shared_lm_head': self.shared_lm_head}

    def checkpoint_buffers(self):
        return {}

    def validate_weight(self, name, value):
        pass

    def validate_checkpoint(self):
        pass

    def attach_head(self, target_head):
        pass


class DFlashDraft(BlockDraftBase):
    shared_lm_head = True
    sample_from_anchor = False

    @staticmethod
    def target_feature_layers(raw):
        return tuple(i+1 for i in raw['dflash_config']['target_layer_ids'])

    def __init__(self, raw, target_hidden, target_vocab, feature_count):
        super().__init__(raw, target_hidden, target_vocab, feature_count)
        if raw.get('draft_vocab_size') not in (None, 0, target_vocab):
            raise ValueError('DFlash requires the target vocabulary')

    def candidates(self, hidden, anchor, count):
        return F.linear(hidden[1:count+1], self.lm_head.weight).argmax(-1), None

    def attach_head(self, target_head):
        self.lm_head = target_head


class DSparkDraft(BlockDraftBase):
    shared_lm_head = False

    @staticmethod
    def target_feature_layers(raw):
        return tuple(raw['aux_hidden_state_layer_ids'])

    def __init__(self, raw, target_hidden, target_vocab, feature_count):
        super().__init__(raw, target_hidden, target_vocab, feature_count)
        self.sample_from_anchor = raw.get('sample_from_anchor', False)
        self.draft_vocab_size = raw.get('draft_vocab_size') or target_vocab
        if raw.get('markov_head_type', 'vanilla') != 'vanilla':
            raise ValueError('Only vanilla DSpark Markov head is supported')
        rank = raw['markov_rank']
        self.lm_head = nn.Linear(self.config.hidden_size, self.draft_vocab_size, bias=False)
        self.markov_head = nn.Module()
        self.markov_head.markov_w1 = nn.Embedding(target_vocab, rank)
        self.markov_head.markov_w2 = nn.Linear(rank, self.draft_vocab_size, bias=False)
        self.register_buffer('d2t', torch.zeros(self.draft_vocab_size, dtype=torch.int64))
        if raw.get('enable_confidence_head', False):
            self.confidence_with_markov = raw.get('confidence_head_with_markov', True)
            self.confidence_head = nn.Module()
            self.confidence_head.proj = nn.Linear(self.config.hidden_size+(rank if self.confidence_with_markov else 0), 1, dtype=torch.float32)

    def candidates(self, hidden, anchor, count):
        values = hidden[:count] if self.sample_from_anchor else hidden[1:count+1]
        logits = F.linear(values, self.lm_head.weight)
        tokens, confidences = [], []
        previous = anchor.reshape(1)
        for value, base in zip(values, logits):
            markov = self.markov_head.markov_w1(previous)[0]
            draft_id = (base+self.markov_head.markov_w2(markov)).argmax(-1)
            previous = (draft_id+self.d2t[draft_id]).reshape(1)
            tokens.append(previous)
            if hasattr(self, 'confidence_head'):
                features = torch.cat((value, markov)) if self.confidence_with_markov else value
                confidences.append(self.confidence_head.proj(features.float()).sigmoid())
        return torch.cat(tokens), torch.cat(confidences) if confidences else None

    def checkpoint_buffers(self):
        return {'d2t': self.d2t}

    def validate_weight(self, name, value):
        if name == 'd2t' and value.dtype not in (torch.int32, torch.int64):
            raise ValueError('DSpark d2t must contain integer offsets')

    def validate_checkpoint(self):
        mapping = torch.arange(self.draft_vocab_size, device=self.d2t.device)+self.d2t
        if (mapping.min() < 0 or mapping.max() >= self.config.vocab_size
                or mapping.unique().numel() != mapping.numel()):
            raise ValueError('Invalid block draft vocabulary mapping')
