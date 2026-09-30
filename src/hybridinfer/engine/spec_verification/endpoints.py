"""Original-trial GDN endpoints, selected without replay or length D2H."""
import torch
from hybridinfer.utils.context import get_context


def begin_endpoints(runner, count, requests=None):
    # Bound all layers' snapshots before launching the trial. The existing OOM
    # fallback runs ordinary decode if the configured budget is insufficient.
    size = 0
    for layer in runner.gdn_layers:
        conv = layer.conv_states
        recurrent = layer.recurrent_states
        compact = (requests is not None
                   and getattr(layer, 'decode_backend', None) == 'pool'
                   and recurrent.dtype == torch.float32)
        conv_elements = ((count+requests*conv.shape[-1])*conv.shape[1]
                         if compact else count*conv[0].numel())
        size += (conv_elements*conv.element_size()
                 + count*recurrent[0].numel()*recurrent.element_size())
    budget = getattr(runner.config.speculative, "state_snapshot_budget_mb", 256)*1024**2
    if size > budget:
        raise torch.cuda.OutOfMemoryError(f"verification states require {size} bytes; budget {budget}")
    endpoints = {}
    get_context().state_endpoints = endpoints
    return endpoints


def select_endpoints(endpoints, slots, indices):
    for layer, conv, recurrent in endpoints.values():
        if hasattr(conv, 'commit'):
            conv.commit(layer.conv_states, slots, indices)
        else:
            layer.conv_states.index_copy_(0, slots, conv.index_select(0, indices))
        layer.recurrent_states.index_copy_(0, slots, recurrent.index_select(0, indices))
