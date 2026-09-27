from collections import deque

from hybridinfer.config import Config
from hybridinfer.engine.sequence import Sequence, SequenceStatus
from hybridinfer.engine.block_manager import BlockManager
from hybridinfer.engine.prefix_checkpoint import PrefixCheckpointManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.checkpoints = (PrefixCheckpointManager(getattr(config, "prefix_cache_num_snapshots", 8))
                            if self.enable_prefix_cache and getattr(config, "is_hybrid", False) else None)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # MRV2 zombie equivalent: seqs dispatched to execute_model but not yet
        # consumed by postprocess. They are moved OUT of waiting/running while
        # in flight, so schedule() can never re-dispatch them (their next
        # decode step depends on the in-flight sample).
        self.in_flight: set[int] = set()
        self.resident: set[int] = set()
        self.preempted: list[int] = []

    def is_finished(self):
        return not self.waiting and not self.running and not self.in_flight

    def add(self, seq: Sequence):
        if not 0 < seq.num_tokens <= self.max_model_len:
            raise ValueError("prompt must fit within max_model_len and contain tokens")
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
        # A sequence is either awaiting dispatch (waiting/running) or in
        # flight. Overlap would let the engine dispatch the same request
        # twice and corrupt its KV/GDN state slots.
        ready_ids = {
            seq.seq_id for seq in self.waiting
        } | {
            seq.seq_id for seq in self.running
        }
        assert not (self.in_flight & ready_ids), (
            "scheduler state overlap: seq is both in_flight and ready"
        )

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            # A chunk continuation already owns its KV blocks and slot.
            # Do not let an unadmitted head request block it when the pools
            # are full; that can leave no runnable work able to free memory.
            if self.waiting[0].seq_id not in self.resident:
                continuation = next(
                    (item for item in self.waiting if item.seq_id in self.resident), None,
                )
                if continuation is not None:
                    self.waiting.remove(continuation)
                    self.waiting.appendleft(continuation)
            seq = self.waiting[0]
            if seq.seq_id not in self.resident and len(self.resident) >= self.max_num_seqs:
                break
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # Always check capacity. can_allocate() doubles as the
                # free-block admission check; skipping it when prefix caching
                # is disabled let allocate() pop from an empty deque once the
                # KV pool was exhausted (benchmark bs=112 crash).
                cached = self.block_manager.find_cached_blocks(seq) if self.enable_prefix_cache else 0
                snapshot_id = None
                if self.checkpoints:
                    cached, snapshot_id = self.checkpoints.lookup(seq, cached)
                num_cached_blocks = self.block_manager.can_allocate(seq, cached)
                if num_cached_blocks == -1:
                    if snapshot_id is not None:
                        self.checkpoints.release(snapshot_id)
                    if not scheduled_seqs and not self.running and not self.in_flight:
                        raise RuntimeError(
                            "KV cache exhausted before admission: "
                            f"need {seq.num_blocks} blocks, "
                            f"have {len(self.block_manager.free_block_ids)} free"
                        )
                    break
                seq.restore_snapshot_id = snapshot_id
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                if seq.restore_snapshot_id is not None:
                    self.checkpoints.release(seq.restore_snapshot_id)
                    seq.restore_snapshot_id = None
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            if self.checkpoints:
                start = seq.num_cached_tokens
                end = start + seq.num_scheduled_tokens
                boundary = (seq.num_tokens - 1) // self.block_size * self.block_size
                if start < boundary < end:
                    end = boundary
                elif end < seq.num_tokens and end // self.block_size * self.block_size > start:
                    end = end // self.block_size * self.block_size
                if end % self.block_size == 0:
                    seq.save_snapshot_id = self.checkpoints.reserve(seq, end)
                    # Only split for a checkpoint we can actually save. A full
                    # pinned pool or an existing entry needs no extra forward.
                    if seq.save_snapshot_id is not None:
                        seq.num_scheduled_tokens = end - start
            num_batched_tokens += seq.num_scheduled_tokens
            self.waiting.popleft()
            self.in_flight.add(seq.seq_id)
            self.resident.add(seq.seq_id)
            scheduled_seqs.append(seq)

        # Fill any remaining token budget with decode rows. This produces a
        # mixed prefill+decode batch when a prefill request leaves budget
        # available, and remains a pure decode batch when no prefill was
        # scheduled.
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                self.in_flight.add(seq.seq_id)
                scheduled_seqs.append(seq)
                num_batched_tokens += 1

        any_prefill = any(seq.is_prefill for seq in scheduled_seqs)
        scheduled_ids = [seq.seq_id for seq in scheduled_seqs]
        assert len(set(scheduled_ids)) == len(scheduled_ids), (
            "scheduler produced a duplicate request in one batch"
        )
        return scheduled_seqs, any_prefill

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.in_flight.discard(seq.seq_id)
        self.resident.discard(seq.seq_id)
        self.preempted.append(seq.seq_id)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.in_flight.discard(seq.seq_id)  # sample consumed, seq schedulable again
            if self.enable_prefix_cache:
                self.block_manager.hash_blocks(seq)
            if self.checkpoints:
                if seq.restore_snapshot_id is not None:
                    self.checkpoints.hits += 1
                    self.checkpoints.hit_tokens += seq.num_cached_tokens
                    self.checkpoints.release(seq.restore_snapshot_id)
                    seq.restore_snapshot_id = None
                if seq.save_snapshot_id is not None:
                    self.checkpoints.release(seq.save_snapshot_id, publish=True)
                    seq.save_snapshot_id = None
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                self.waiting.append(seq)  # chunked prefill: back to waiting for the next chunk
                continue
            seq.append_token(token_id)
            if ((not seq.ignore_eos and token_id == self.eos)
                    or seq.num_completion_tokens == seq.max_tokens
                    or seq.num_cached_tokens >= self.max_model_len):
                seq.status = SequenceStatus.FINISHED
                self.resident.discard(seq.seq_id)
                self.block_manager.deallocate(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # back to schedulable
