"""Cache ownership for full attention and GDN groups.

CPU KVCacheManager coordinates prefix intersection, allocation and publication.
KVCacheStorage owns per-worker GPU payloads; execution kernels consume its views.
Keep CPU ownership separate from GPU storage for tensor-parallel workers.
"""
from collections import deque
from dataclasses import dataclass
import torch
import xxhash
import numpy as np

from hybridinfer.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def find_cached_blocks(self, seq: Sequence) -> int:
        h = -1
        count = 0
        # Keep at least one token to compute logits, even for exact matches.
        for i in range((seq.num_tokens - 1) // self.block_size):
            tokens = seq.block(i)
            h = self.compute_hash(tokens, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != tokens:
                break
            count += 1
        return count

    def can_allocate(self, seq: Sequence, num_cached_blocks: int | None = None) -> int:
        if num_cached_blocks is None:
            num_cached_blocks = self.find_cached_blocks(seq)
        h = -1
        shared_used = 0
        for i in range(num_cached_blocks):
            h = self.compute_hash(seq.block(i), h)
            shared_used += self.hash_to_block_id[h] in self.used_block_ids
        if len(self.free_block_ids) < seq.num_blocks - shared_used:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def reserve_trial(self, seq: Sequence, computed_end: int) -> bool:
        """Reserve unpublished tail pages without changing committed history.

        Shared writable tails conservatively fall back until copy-on-write is
        implemented. Shared complete prefix pages are never trial write targets.
        """
        if computed_end < seq.num_cached_tokens:
            raise ValueError("trial endpoint precedes committed state")
        first = seq.num_cached_tokens // self.block_size
        for block_id in seq.block_table[first:]:
            if self.blocks[block_id].ref_count != 1:
                return False
        count = (computed_end + self.block_size - 1) // self.block_size
        needed = max(0, count - len(seq.block_table))
        if needed > len(self.free_block_ids):
            return False
        for _ in range(needed):
            seq.block_table.append(self._allocate_block())
        return True

    def trim_trial(self, seq: Sequence, computed_end: int):
        count = (computed_end + self.block_size - 1) // self.block_size
        while len(seq.block_table) > count:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id



@dataclass
class Checkpoint:
    snapshot_id: int
    tokens: tuple[int, ...]
    pins: int = 1
    ready: bool = False
    touched: int = 0


class PrefixCheckpointManager:
    """GDN group metadata; readers/writers stay pinned until completion."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.clock = 0
        self.hits = 0
        self.hit_tokens = 0

    def key(self, seq, length):
        h = -1
        for i in range(length // seq.block_size):
            h = BlockManager.compute_hash(seq.block(i), h)
        return h

    def lookup(self, seq, max_blocks):
        hashes = []
        h = -1
        for i in range(max_blocks):
            h = BlockManager.compute_hash(seq.block(i), h)
            hashes.append(h)
        for blocks in range(max_blocks, 0, -1):
            length = blocks * seq.block_size
            entry = self.entries.get(hashes[blocks - 1])
            if entry and entry.ready and entry.tokens == tuple(seq.token_ids[:length]):
                entry.pins += 1
                self.clock += 1
                entry.touched = self.clock
                return blocks, entry.snapshot_id
        return 0, None

    def reserve(self, seq, length):
        key = self.key(seq, length)
        if key in self.entries:
            return None
        used = {e.snapshot_id for e in self.entries.values()}
        available = next((i for i in range(self.capacity) if i not in used), None)
        if available is None:
            victims = [(k, e) for k, e in self.entries.items() if e.pins == 0]
            if not victims:
                return None
            victim_key, victim = min(victims, key=lambda item: item[1].touched)
            available = victim.snapshot_id
            del self.entries[victim_key]
        self.clock += 1
        self.entries[key] = Checkpoint(available, tuple(seq.token_ids[:length]),
                                       touched=self.clock)
        return available

    def release(self, snapshot_id, publish=False):
        entry = next(e for e in self.entries.values() if e.snapshot_id == snapshot_id)
        assert entry.pins > 0
        entry.pins -= 1
        if publish:
            entry.ready = True


class KVCacheManager:
    """Scheduler-facing facade coordinating the attention and GDN cache groups."""
    def __init__(self, config):
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.block_pool = BlockManager(config.num_kvcache_blocks, self.block_size)
        self.checkpoints = (PrefixCheckpointManager(getattr(config, "prefix_cache_num_snapshots", 8))
                            if config.enable_prefix_cache and getattr(config, "is_hybrid", False) else None)

    def get_computed_blocks(self, seq):
        cached = self.block_pool.find_cached_blocks(seq) if self.enable_prefix_cache else 0
        snapshot = None
        if self.checkpoints:
            cached, snapshot = self.checkpoints.lookup(seq, cached)
        seq.restore_snapshot_id = snapshot
        return cached

    def release_reader(self, seq):
        if seq.restore_snapshot_id is not None:
            self.checkpoints.release(seq.restore_snapshot_id)
            seq.restore_snapshot_id = None

    def plan_prefill(self, seq):
        """Reserve a state checkpoint and align the forward to its boundary."""
        if not self.checkpoints:
            return
        start = seq.num_cached_tokens
        end = start + seq.num_scheduled_tokens
        boundary = (seq.num_tokens - 1) // self.block_size * self.block_size
        if start < boundary < end:
            end = boundary
        elif end < seq.num_tokens and end // self.block_size * self.block_size > start:
            end = end // self.block_size * self.block_size
        if end % self.block_size == 0:
            seq.save_snapshot_id = self.checkpoints.reserve(seq, end)
            if seq.save_snapshot_id is not None:
                seq.num_scheduled_tokens = end - start

    def cache_blocks(self, seq):
        if self.enable_prefix_cache:
            self.block_pool.hash_blocks(seq)

    def complete(self, seq):
        self.cache_blocks(seq)
        if self.checkpoints:
            if seq.restore_snapshot_id is not None:
                self.checkpoints.hits += 1
                self.checkpoints.hit_tokens += seq.num_cached_tokens
                self.release_reader(seq)
            if seq.save_snapshot_id is not None:
                self.checkpoints.release(seq.save_snapshot_id, publish=True)
                seq.save_snapshot_id = None

    def free(self, seq):
        self.release_reader(seq)
        self.block_pool.deallocate(seq)

    def can_allocate(self, seq, cached):
        return self.block_pool.can_allocate(seq, cached)

    def allocate_slots(self, seq, cached):
        self.block_pool.allocate(seq, cached)

    def can_append(self, seq):
        return self.block_pool.can_append(seq)

    def append_slots(self, seq):
        self.block_pool.may_append(seq)

    def reserve_trial(self, seq, end):
        return self.block_pool.reserve_trial(seq, end)

    def trim_trial(self, seq, end):
        self.block_pool.trim_trial(seq, end)


class KVCacheStorage:
    """Per-worker GPU allocation and GDN checkpoint copying."""
    def __init__(self, config, model, gdn_layers):
        self.config = config
        self.model = model
        self.gdn_layers = gdn_layers
        self.block_size = config.kvcache_block_size
        self.kv_cache = None
        self.prefix_snapshots = []

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.config.tensor_parallel_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)

        # Only full_attention layers hold K/V cache; GDN layers keep their own
        # conv/recurrent state pools instead. Count cache-bearing modules from
        # the model structure (robust to both ``layer_types`` and the
        # ``full_attention_interval`` fallback, and to Dense checkpoints where
        # every layer is attention).
        num_attn_layers = sum(
            1 for module in self.model.modules()
            if hasattr(module, "k_cache") and hasattr(module, "v_cache")
        )
        assert num_attn_layers > 0, "no attention layers found; cannot size KV cache"
        block_bytes = (
            2 * num_attn_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * hf_config.dtype.itemsize
        )
        snapshot_reserve = (config.speculative.state_snapshot_budget_mb*1024**2
                            if config.speculative and config.speculative.enabled else 0)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current
                                       - snapshot_reserve) // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise ValueError("GPU memory budget cannot fit target KV after state snapshot reserve; "
                             "reduce state_snapshot_budget_mb or request capacity")
        self.kv_cache = torch.empty(
            2, num_attn_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_attn_layers, (
            f"attention layer drift: assigned {layer_id} caches but sized "
            f"for {num_attn_layers}"
        )

    def allocate_gdn_state_pool(self):
        num_slots = self.config.max_num_seqs
        speculative = self.config.speculative
        if speculative and speculative.enabled:
            num_slots *= 2  # One private trial slot per active verification request.
        for layer in self.gdn_layers:
            layer.allocate_state_pool(num_slots)

    def allocate_prefix_snapshots(self):
        count = self.config.prefix_cache_num_snapshots if self.config.enable_prefix_cache else 0
        self.prefix_snapshots = [
            (layer.conv_states.new_empty((count, *layer.conv_states.shape[1:])),
             layer.recurrent_states.new_empty((count, *layer.recurrent_states.shape[1:])))
            for layer in self.gdn_layers
        ]

    def copy_prefix_state(self, seq, slot, restore):
        snapshot_id = seq.restore_snapshot_id if restore else seq.save_snapshot_id
        if snapshot_id is None:
            return
        for layer, (conv, recurrent) in zip(self.gdn_layers, self.prefix_snapshots):
            for runtime, cached in ((layer.conv_states, conv), (layer.recurrent_states, recurrent)):
                if restore:
                    runtime[slot].copy_(cached[snapshot_id])
                else:
                    cached[snapshot_id].copy_(runtime[slot])


    def prepare_requests(self, new_entries, device):
        if not new_entries or not self.gdn_layers:
            return
        from .staged_write import to_device
        slots = to_device([slot for _, slot in new_entries], torch.int64, device)
        for layer in self.gdn_layers:
            layer.reset_state(slots)
        for seq, slot in new_entries:
            self.copy_prefix_state(seq, slot, restore=True)

    def save_requests(self, seqs, slots):
        for seq, slot in zip(seqs, slots):
            self.copy_prefix_state(seq, slot, restore=False)

    def clear(self):
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = module.v_cache = torch.tensor([])
        for layer in self.gdn_layers:
            layer.conv_states = layer.recurrent_states = torch.tensor([])
        self.kv_cache = None
        self.prefix_snapshots.clear()
        self.model = None
        self.gdn_layers = []
