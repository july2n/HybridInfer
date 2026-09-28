"""Checkpoint-backed Qwen3.5 MTP with shared embedding/head and private KV."""
from copy import deepcopy
from glob import glob
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from hybridinfer.layers.layernorm import GemmaRMSNorm
from hybridinfer.models.qwen3_5 import Qwen3_5DecoderLayer


class Qwen3_5MTP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if getattr(config, 'mtp_num_hidden_layers', 1) != 1:
            raise ValueError('Initial Qwen3.5 MTP backend requires exactly one MTP layer')
        if getattr(config, 'mtp_use_dedicated_embeddings', False):
            raise ValueError('Dedicated MTP embeddings are not supported')
        cfg = deepcopy(config)
        cfg.layer_types = ['full_attention']
        cfg.num_hidden_layers = 1
        self.fc = nn.Linear(cfg.hidden_size*2, cfg.hidden_size, bias=False)
        self.pre_fc_norm_embedding = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(cfg, 0)])
        self.norm = GemmaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, embeddings, target_hidden, positions):
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(embeddings),
                                   self.pre_fc_norm_hidden(target_hidden)), -1))
        hidden, residual = self.layers[0](positions, hidden, None)
        hidden, _ = self.norm(hidden, residual)
        return hidden

    def load_checkpoint(self, path):
        params = dict(self.named_parameters())
        loaded, shards, source = set(), {}, []
        for filename in sorted(glob(str(Path(path)/'*.safetensors'))):
            with safe_open(filename, framework='pt', device='cpu') as f:
                for original in f.keys():
                    if not original.startswith(('mtp.', 'model.mtp.')):
                        continue
                    name = original.split('mtp.', 1)[1]
                    value = f.get_tensor(original)
                    shard = None
                    for part, index in [('gate_proj', 0), ('up_proj', 1)]:
                        if part in name.split('.'):
                            name = name.replace(part, 'gate_up_proj')
                            shard = index
                            break
                    if name not in params:
                        raise ValueError(f'Unexpected MTP weight: {original}')
                    param = params[name]
                    if shard is None:
                        if param.shape != value.shape:
                            raise ValueError(f'MTP weight shape mismatch: {original}')
                        param.data.copy_(value)
                        loaded.add(name)
                    else:
                        param.weight_loader(param, value, shard)
                        shards.setdefault(name, set()).add(shard)
                    source.append(original)
        for name, indices in shards.items():
            if indices != {0, 1}:
                raise ValueError(f'Incomplete MTP packed weight: {name}')
            loaded.add(name)
        if loaded != set(params):
            raise ValueError(f'Missing MTP weights: {sorted(set(params)-loaded)}')
        return {'checkpoint_weights': source, 'loaded_parameters': len(loaded)}
