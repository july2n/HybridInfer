from collections import deque

from hybridinfer.config import Config
from hybridinfer.engine.sequence import Sequence, SequenceStatus
from hybridinfer.engine.kv_cache_manager import KVCacheManager


class Scheduler:

    def __init__(self, config: Config):
        self.speculative = getattr(config, "speculative", None)
        self.draft_proposer = None
        self.spec_fallbacks = {}
        self._spec_trial_block_counts = {}
        self.max_num_seqs = config.max_num_seqs
        self.max_model_len = config.max_model_len
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_speculative_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.enable_prefix_cache = config.enable_prefix_cache
        self.kv_cache_manager = KVCacheManager(config)
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

    def begin_speculative_batch(self):
        """Reserve ready decode requests, including B=1 and zero-draft rows."""
        config = self.speculative
        if not config or not config.enabled:
            return None
        def fallback(reason):
            self.spec_fallbacks[reason] = self.spec_fallbacks.get(reason, 0)+1
            return None
        if self.waiting or self.in_flight or not self.running:
            return fallback("batch_or_prefill")
        from hybridinfer.spec_decode.interfaces import DraftContext, VerificationPlan
        from hybridinfer.spec_decode.ngram import NgramProposer
        seqs = list(self.running)[:min(self.max_num_seqs, self.max_num_batched_tokens, self.max_speculative_tokens)]
        if any(seq.temperature != 0 for seq in seqs) and config.verification_mode != "packed":
            return fallback("temperature")
        remaining = self.max_speculative_tokens
        plans = []
        proposer = self.draft_proposer or NgramProposer(config)
        for row, seq in enumerate(seqs):
            # Reserve at least one anchor row for each later request.
            budget = remaining-(len(seqs)-row-1)
            context = DraftContext(seq.seq_id, tuple(seq.token_ids), seq.num_cached_tokens,
                                   seq.max_tokens-seq.num_completion_tokens,
                                   self.max_model_len, budget)
            proposal = proposer.propose([context])
            candidates = proposal.tokens_for(0)
            plan = VerificationPlan(seq.seq_id, seq.num_cached_tokens, seq.last_token, candidates,
                                    proposal.probabilities)
            plans.append(plan)
            remaining -= len(plan.input_tokens)
        if not any(p.candidates for p in plans):
            return fallback("no_draft_or_budget")
        original_counts = {seq.seq_id: len(seq.block_table) for seq in seqs}
        for seq, plan in zip(seqs, plans):
            if not self.kv_cache_manager.reserve_trial(seq, plan.trial_end):
                for reserved in seqs:
                    self.kv_cache_manager.trim_trial(reserved, original_counts[reserved.seq_id]*self.block_size)
                return fallback("kv_capacity_or_shared_tail")
        self._spec_trial_block_counts.update(original_counts)
        for seq in seqs:
            self.running.remove(seq)
            self.in_flight.add(seq.seq_id)
        return seqs, tuple(plans)

    def abort_speculative_batch(self, seqs):
        for seq in reversed(seqs):
            original = self._spec_trial_block_counts.pop(seq.seq_id)
            self.kv_cache_manager.trim_trial(seq, original*self.block_size)
            self.in_flight.discard(seq.seq_id)
            self.running.appendleft(seq)

    def finish_speculative(self, seq, result):
        if seq.seq_id not in self.in_flight:
            raise RuntimeError("speculative request is not in flight")
        self._spec_trial_block_counts.pop(seq.seq_id, None)
        seq.append_tokens(result.token_ids)
        seq.num_scheduled_tokens = result.committed_computed_length - seq.num_cached_tokens
        self.kv_cache_manager.trim_trial(seq, result.committed_computed_length)
        self.kv_cache_manager.cache_blocks(seq)
        seq.num_cached_tokens = result.committed_computed_length
        seq.num_scheduled_tokens = 0
        self.in_flight.remove(seq.seq_id)
        if result.finished:
            seq.status = SequenceStatus.FINISHED
            self.resident.discard(seq.seq_id)
            self.kv_cache_manager.free(seq)
        else:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

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
                cached = self.kv_cache_manager.get_computed_blocks(seq)
                num_cached_blocks = self.kv_cache_manager.can_allocate(seq, cached)
                if num_cached_blocks == -1:
                    self.kv_cache_manager.release_reader(seq)
                    if not scheduled_seqs and not self.running and not self.in_flight:
                        raise RuntimeError(
                            "KV cache exhausted before admission: "
                            f"need {seq.num_blocks} blocks, "
                            f"have {len(self.block_manager.free_block_ids)} free"
                        )
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                self.kv_cache_manager.release_reader(seq)
                break
            if not seq.block_table:
                self.kv_cache_manager.allocate_slots(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            seq.is_prefill = True
            self.kv_cache_manager.plan_prefill(seq)
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
            while not self.kv_cache_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.kv_cache_manager.append_slots(seq)
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
        self.kv_cache_manager.free(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.in_flight.discard(seq.seq_id)  # sample consumed, seq schedulable again
            self.kv_cache_manager.complete(seq)
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
                self.kv_cache_manager.free(seq)
            else:
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)  # back to schedulable

    @property
    def block_manager(self):
        return self.kv_cache_manager.block_pool

    @property
    def checkpoints(self):
        return self.kv_cache_manager.checkpoints
