"""Persistent request slots, independent of each step's input ordering."""
from collections import deque

import torch

from .staged_write import StagedWriteTensor


class InputBatch:
    def __init__(self, max_num_seqs):
        self.seqs = [None] * max_num_seqs
        self.seq_id_to_slot = {}
        self._free = deque(range(max_num_seqs))

    def update(self, seqs):
        new_entries = []
        if len({s.seq_id for s in seqs}) != len(seqs):
            raise ValueError("duplicate request in batch")
        needed = sum(s.seq_id not in self.seq_id_to_slot for s in seqs)
        if needed > len(self._free):
            raise RuntimeError("persistent request slot capacity exhausted")
        for seq in seqs:
            slot = self.seq_id_to_slot.get(seq.seq_id)
            if slot is None:
                slot = self._free.popleft()
                self.seq_id_to_slot[seq.seq_id] = slot
                new_entries.append((seq, slot))
            self.seqs[slot] = seq
        return seqs, new_entries

    def slots_for(self, seqs):
        return [self.seq_id_to_slot[s.seq_id] for s in seqs]

    def remove_finished(self):
        for seq_id, slot in list(self.seq_id_to_slot.items()):
            if self.seqs[slot].is_finished:
                self.remove(seq_id)

    def remove(self, seq_id):
        slot = self.seq_id_to_slot.pop(seq_id, None)
        if slot is not None:
            self.seqs[slot] = None
            self._free.append(slot)

    def clear(self):
        self.seqs[:] = [None] * len(self.seqs)
        self.seq_id_to_slot.clear()
        self._free = deque(range(len(self.seqs)))


class RequestState:
    """GPU-owned tokens, computed lengths, block tables and sampling state."""
    def __init__(self, config, device="cuda"):
        n = config.max_num_seqs
        # One extra cell holds the final sampled token; it is never forwarded
        # beyond max_model_len.
        self.capacity = config.max_model_len + 1
        blocks = (config.max_model_len + config.kvcache_block_size - 1) // config.kvcache_block_size
        self.tokens = StagedWriteTensor((n, self.capacity), torch.int64, device)
        self.block_tables = StagedWriteTensor((n, blocks), torch.int32, device, fill=-1)
        self.computed = StagedWriteTensor((n,), torch.int32, device)
        self.temperatures = StagedWriteTensor((n,), torch.float32, device)
        self.seeds = StagedWriteTensor((n,), torch.int64, device)
        self._blocks = {}

    def update(self, seqs, slots, new_entries):
        new_slots = {slot for _, slot in new_entries}
        for seq, slot in zip(seqs, slots):
            if slot in new_slots:
                self.tokens.stage_write(slot, 0, seq.token_ids)
                self.computed.stage_write(slot, 0, [seq.num_cached_tokens])
                self.temperatures.stage_write(slot, 0, [seq.temperature])
                seed = seq.seed if seq.seed is not None else (torch.initial_seed() + seq.seq_id) % (2**63)
                self.seeds.stage_write(slot, 0, [seed])
                self._blocks[slot] = ()
                # Remove stale blocks when a freed slot is reused.
                self.block_tables.stage_write(slot, 0, [-1] * self.block_tables.tensor.shape[1])
            old = self._blocks[slot]
            current = tuple(seq.block_table)
            first = next((i for i, (a, b) in enumerate(zip(old, current)) if a != b), min(len(old), len(current)))
            self.block_tables.stage_write(slot, first, current[first:])
            if len(current) < len(old):
                self.block_tables.stage_write(slot, len(current), [-1] * (len(old) - len(current)))
            self._blocks[slot] = current
        for state in (self.tokens, self.block_tables, self.computed, self.temperatures, self.seeds):
            state.apply_write()

    def remove(self, slot):
        self._blocks.pop(slot, None)
