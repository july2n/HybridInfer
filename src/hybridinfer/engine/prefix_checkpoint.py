"""CPU checkpoint ownership; GPU payloads live on each model runner.

Reservations and readers stay pinned through batch completion. Pending writes
are invisible to lookup, including when other batches are dispatched first.
"""
from dataclasses import dataclass

from .block_manager import BlockManager


@dataclass
class Checkpoint:
    snapshot_id: int
    tokens: tuple[int, ...]
    pins: int = 1
    ready: bool = False
    touched: int = 0


class PrefixCheckpointManager:
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
