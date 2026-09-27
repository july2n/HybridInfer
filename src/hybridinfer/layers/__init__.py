"""Load individual kernels only when requested."""
from importlib import import_module

_EXPORT_MODULES = {
    'SiluAndMul': 'activation',
    'Attention': 'attention',
    'store_kvcache': 'attention',
    'ParallelLMHead': 'embed_head',
    'VocabParallelEmbedding': 'embed_head',
    'GatedDeltaNet': 'gated_delta_net',
    'chunk_gated_delta_rule': 'gated_delta_net',
    'decode_gated_delta_rule': 'gated_delta_net',
    'GemmaRMSNorm': 'layernorm',
    'RMSNorm': 'layernorm',
    'RMSNormGated': 'layernorm',
    'ColumnParallelLinear': 'linear',
    'LinearBase': 'linear',
    'MergedColumnParallelLinear': 'linear',
    'QKVParallelLinear': 'linear',
    'ReplicatedLinear': 'linear',
    'RowParallelLinear': 'linear',
    'InterleavedMRoPE': 'rotary_embedding',
    'RotaryEmbedding': 'rotary_embedding',
    'apply_rotary_emb': 'rotary_embedding',
    'get_rope': 'rotary_embedding',
    'Sampler': 'sampler',
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    if name not in _EXPORT_MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORT_MODULES[name]}"), name)
    globals()[name] = value
    return value
