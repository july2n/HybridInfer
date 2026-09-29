# KV cache ownership

`engine/kv_cache_manager.py` owns cache allocation and lifecycle. The interface
follows vLLM's separation between scheduler-facing cache management, per-type
cache ownership, and worker-side tensor storage:
https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/

- `KVCacheManager`: scheduler-facing prefix lookup, allocation, append/free,
  speculative tail reservations, checkpoint planning and publication.
- `BlockManager`: full-attention physical blocks, chained prefix hashes and
  reference counts.
- `PrefixCheckpointManager`: GDN boundary checkpoints, reader/writer pins,
  readiness and LRU eviction.
- `KVCacheStorage`: per-TP-worker attention tensors, GDN runtime pools and
  checkpoint payloads; resets, restores, saves and clears GPU storage.

Prefix lookup first finds a continuous attention prefix, then selects the
longest ready GDN checkpoint within that prefix. A GDN snapshot alone or an
attention-only hit cannot skip computation in a hybrid model. Pins survive
asynchronous dispatch until scheduler postprocessing publishes/releases them.

The runner allocates GDN storage before warmup, sizes attention storage after
warmup, and invokes request preparation/save around forward execution. It no
longer implements memory budgeting, tensor allocation or checkpoint copying.
The scheduler owns scheduling budgets, while `plan_prefill` selects checkpoint
boundaries and adjusts the scheduled token count.

This refactor retains the existing attention block layout and bounded GDN
checkpoint capacity (`prefix_cache_num_snapshots`); it does not adopt vLLM's
uniform padded page layout or expose arbitrary cache-group configurations.
Snapshots are selected at prefill chunk boundaries and the final reusable
prompt boundary, rather than at every block boundary. Old block/checkpoint
modules and runner tensor properties remain compatibility accessors.

Validation:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python benchmarks/validate_prefix_cache.py --json-out /tmp/prefix.json
PYTHONPATH=src python benchmarks/validate_prefix_cache.py --graphs --json-out /tmp/prefix_graphs.json
```
