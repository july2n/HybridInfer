"""Shared model-based proposal lifecycle, feature history and draft KV cache."""
import torch
from hybridinfer.utils.context import (set_context, get_context, reset_context,
                                       BatchDescriptor)
from .interfaces import DeviceDraftProposal
from hybridinfer.sampling.rejection_sampler import categorical, random_uniform


class SpecDecodeBaseProposer:
    """Common lifecycle for model-backed draft proposers.

    Subclasses load ``self.model`` and call ``_initialize_cache`` after this
    constructor. Block models override ``propose_device``; linear MTP/EAGLE
    models use the implementation below.
    """

    def __init__(self, runner, *, pass_hidden_states_to_model=False):
        self.runner = runner
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self._target_hidden = None

    def _initialize_cache(self, feature_size):
        runner = self.runner
        cfg = runner.config
        device = runner.model.lm_head.weight.device
        dtype = runner.model.lm_head.weight.dtype
        self.features = torch.empty((cfg.max_num_seqs, cfg.max_model_len,
                                     feature_size), dtype=dtype, device=device)
        self.validated = [0]*cfg.max_num_seqs
        self.owners = [None]*cfg.max_num_seqs
        self.last_hidden = {}
        self.pages = (cfg.max_model_len+cfg.kvcache_block_size-1)//cfg.kvcache_block_size
        for layer in self.model.layers:
            attn = layer.self_attn.attn
            shape = (cfg.max_num_seqs*self.pages, cfg.kvcache_block_size,
                     attn.num_kv_heads, attn.head_dim)
            attn.k_cache = torch.empty(shape, dtype=dtype, device=device)
            attn.v_cache = torch.empty_like(attn.k_cache)

    def forward_target(self, ids, positions):
        if not self.pass_hidden_states_to_model:
            return self.runner.model(ids, positions)
        context = get_context()
        previous = context.feature_layers, context.target_features
        features = {}
        context.feature_layers, context.target_features = self.feature_layers, features
        self._target_hidden = None
        try:
            hidden = self.runner.model(ids, positions)
            self._target_hidden = self.model.combine_features(
                [features[i] for i in self.feature_layers])
            return hidden
        finally:
            context.feature_layers, context.target_features = previous

    def _embed_tokens(self, ids):
        return self.runner.model.model.embed_tokens(ids)

    def _compute_logits(self, hidden):
        return self.runner.model.compute_logits(hidden)

    def record(self, slots, positions, hidden, slices=None):
        if self.pass_hidden_states_to_model:
            hidden, self._target_hidden = self._target_hidden, None
            if hidden is None:
                raise RuntimeError('Target features were not captured')
        if slices is None:
            self.features[slots, positions] = hidden
        else:
            for slot, (start, end) in zip(slots, slices):
                self.features[slot, positions[start:end]] = hidden[start:end]

    def release(self, slot):
        self.validated[slot] = 0
        self.owners[slot] = None
        self.last_hidden.pop(slot, None)

    def _claim_slot(self, request_id):
        slot = self.runner.input_batch.seq_id_to_slot[request_id]
        if self.owners[slot] != request_id:
            self.release(slot)
            self.owners[slot] = request_id
        return slot

    def _draft_count(self, context, *, capacity=None):
        limit = self.runner.config.speculative.max_draft_tokens
        if capacity is not None:
            limit = min(limit, capacity)
        return max(0, min(limit, context.remaining_output_tokens-1,
                          context.max_model_len-context.computed_length-1,
                          context.verification_budget-1))

    def _forward(self, slot, ids, target_hidden, positions, physical, length, prefill):
        runner = self.runner
        device = ids.device
        block = runner.block_size
        table = torch.arange(slot*self.pages, (slot+1)*self.pages,
                             device=device, dtype=torch.int32)[None]
        mapping = slot*self.pages*block+physical
        kwargs = dict(slot_mapping=mapping.to(torch.int32), block_tables=table,
                      batch_descriptor=BatchDescriptor('draft', ids.numel(), 1, ids.numel(), ids.numel()))
        if prefill:
            kwargs.update(cu_seqlens_q=torch.tensor([0, ids.numel()], device=device, dtype=torch.int32),
                          cu_seqlens_k=torch.tensor([0, length], device=device, dtype=torch.int32),
                          max_seqlen_q=ids.numel(), max_seqlen_k=length)
        else:
            kwargs.update(context_lens=torch.tensor([length], device=device, dtype=torch.int32))
        set_context(prefill, **kwargs)
        return self.model(self._embed_tokens(ids), target_hidden, positions)

    @torch.inference_mode()
    def propose_device(self, contexts):
        runner = self.runner
        rows, offsets, ids, probabilities = [], [0], [], []
        probabilistic = getattr(runner.config.speculative, 'mtp_draft_sampling', 'greedy') == 'random'
        try:
            for ctx in contexts:
                slot = self._claim_slot(ctx.request_id)
                count = self._draft_count(ctx)
                if count:
                    end = ctx.computed_length
                    start = self.validated[slot]
                    if start > end:
                        raise RuntimeError('Draft cache is ahead of committed target')
                    tokens = runner.request_state.tokens.tensor
                    if start < end:
                        physical = torch.arange(start, end, device=tokens.device)
                        hidden = self._forward(slot, tokens[slot, physical+1],
                            self.features[slot, physical], physical, physical, end, True)
                        self.last_hidden[slot] = hidden[-1:].clone()
                        self.validated[slot] = end
                    hidden = self.last_hidden[slot]
                    drafts = []
                    for step in range(count):
                        logits = self._compute_logits(hidden)
                        if probabilistic:
                            if ctx.temperature == 0:
                                token = logits.argmax(-1)
                                q = torch.zeros_like(logits, dtype=torch.float32)
                                q.scatter_(1, token[:, None], 1.)
                            else:
                                q = torch.softmax(logits.float()/ctx.temperature, -1)
                                seed = ctx.seed if ctx.seed is not None else (torch.initial_seed()+ctx.request_id) % (2**63)
                                token = categorical(q[0], random_uniform(seed, end+step+1, 'draft', logits.device)).reshape(1)
                            probabilities.append(q)
                        else:
                            token = logits.argmax(-1)
                        drafts.append(token)
                        if step+1 < count:
                            position = torch.tensor([end+step], device=tokens.device)
                            hidden = self._forward(slot, token, hidden, position,
                                                   position, end+step+1, False)
                    # Candidates remain on device until the scheduler adapter.
                    row = torch.cat(drafts)
                else:
                    row = torch.empty(0, device=runner.model.lm_head.weight.device, dtype=torch.int64)
                rows.append(row)
                offsets.append(offsets[-1]+row.numel())
                ids.append(ctx.request_id)
            q = None
            if probabilistic:
                q = (torch.cat(probabilities) if probabilities else torch.empty(
                    (0, runner.model.lm_head.weight.shape[0]), dtype=torch.float32,
                    device=runner.model.lm_head.weight.device))
            tokens = torch.cat(rows) if rows else torch.empty(0, dtype=torch.int64, device=runner.model.lm_head.weight.device)
            return DeviceDraftProposal(tuple(ids), tokens, tuple(offsets), q)
        finally:
            reset_context()

    def propose(self, contexts):
        return self.propose_device(contexts).to_host()
