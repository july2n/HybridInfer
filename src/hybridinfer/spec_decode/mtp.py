"""Device feature history and draft-cache lifecycle for linear MTP proposals."""
import torch
from hybridinfer.utils.context import set_context, reset_context, BatchDescriptor
from .interfaces import DeviceDraftProposal, BackendCapabilities
from hybridinfer.models.qwen3_5_mtp import Qwen3_5MTP


class MTPProposer:
    capabilities = BackendCapabilities(requires_target_features=True, supports_random_sampling=True)

    def __init__(self, runner):
        self.runner = runner
        cfg = runner.config
        self.model = Qwen3_5MTP(cfg.hf_config)
        self.weight_report = self.model.load_checkpoint(cfg.model)
        device = runner.model.lm_head.weight.device
        dtype = runner.model.lm_head.weight.dtype
        self.features = torch.empty((cfg.max_num_seqs, cfg.max_model_len,
                                     cfg.hf_config.hidden_size), dtype=dtype, device=device)
        self.validated = [0]*cfg.max_num_seqs
        self.owners = [None]*cfg.max_num_seqs
        self.last_hidden = {}
        self.pages = (cfg.max_model_len+cfg.kvcache_block_size-1)//cfg.kvcache_block_size
        attn = self.model.layers[0].self_attn.attn
        shape = (cfg.max_num_seqs*self.pages, cfg.kvcache_block_size,
                 attn.num_kv_heads, attn.head_dim)
        attn.k_cache = torch.empty(shape, dtype=dtype, device=device)
        attn.v_cache = torch.empty_like(attn.k_cache)

    def record(self, slots, positions, hidden, slices=None):
        if slices is None:
            self.features[slots, positions] = hidden
        else:
            for slot, (start, end) in zip(slots, slices):
                self.features[slot, positions[start:end]] = hidden[start:end]

    def release(self, slot):
        self.validated[slot] = 0
        self.owners[slot] = None
        self.last_hidden.pop(slot, None)

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
        return self.model(runner.model.model.embed_tokens(ids), target_hidden, positions)

    @torch.inference_mode()
    def propose_device(self, contexts):
        runner = self.runner
        rows, offsets, ids = [], [0], []
        try:
            for ctx in contexts:
                slot = runner.input_batch.seq_id_to_slot[ctx.request_id]
                if self.owners[slot] != ctx.request_id:
                    self.release(slot)
                    self.owners[slot] = ctx.request_id
                count = min(runner.config.speculative.max_draft_tokens,
                            ctx.remaining_output_tokens-1,
                            ctx.max_model_len-ctx.computed_length-1,
                            ctx.verification_budget-1)
                count = max(0, count)
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
                        token = runner.model.compute_logits(hidden).argmax(-1)
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
            return DeviceDraftProposal(tuple(ids), torch.cat(rows), tuple(offsets))
        finally:
            reset_context()

    def propose(self, contexts):
        return self.propose_device(contexts).to_host()
