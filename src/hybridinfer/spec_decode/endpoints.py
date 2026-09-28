"""Original-trial GDN endpoints, selected without replay or length D2H."""
import torch
from hybridinfer.utils.context import get_context


def begin_endpoints(runner, count):
    # Bound all layers' snapshots before launching the trial. The existing OOM
    # fallback runs ordinary decode if the configured budget is insufficient.
    size = sum((layer.conv_states[0].numel()*layer.conv_states.element_size()
                + layer.recurrent_states[0].numel()*layer.recurrent_states.element_size())
               for layer in runner.gdn_layers) * count
    budget = getattr(runner.config.speculative, "state_snapshot_budget_mb", 256)*1024**2
    if size > budget:
        raise torch.cuda.OutOfMemoryError(f"verification states require {size} bytes; budget {budget}")
    endpoints = {}
    get_context().state_endpoints = endpoints
    return endpoints


def select_endpoints(endpoints, slots, indices):
    for layer, conv, recurrent in endpoints.values():
        layer.conv_states.index_copy_(0, slots, conv.index_select(0, indices))
        layer.recurrent_states.index_copy_(0, slots, recurrent.index_select(0, indices))
