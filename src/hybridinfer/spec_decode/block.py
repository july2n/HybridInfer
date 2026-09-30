"""Block proposals; only committed target features enter the private KV cache."""
import json
from pathlib import Path
import torch
from hybridinfer.models.block_draft import DFlashDraft, DSparkDraft
from .llm_base_proposer import SpecDecodeBaseProposer
from .interfaces import BackendCapabilities, DeviceDraftProposal


class BlockProposer(SpecDecodeBaseProposer):
    capabilities = BackendCapabilities(requires_target_features=True)

    def __init__(self, runner):
        super().__init__(runner, pass_hidden_states_to_model=True)
        cfg, spec = runner.config, runner.config.speculative
        raw = json.loads((Path(spec.draft_model)/'config.json').read_text())
        model_type = DFlashDraft if spec.method == 'dflash' else DSparkDraft
        selected = model_type.target_feature_layers(raw)
        if (not selected or len(set(selected)) != len(selected)
                or any(type(i) is not int or not 0 <= i < cfg.hf_config.num_hidden_layers for i in selected)):
            raise ValueError('Invalid block draft target feature boundaries')
        self.feature_layers = selected
        self.model = model_type(raw, cfg.hf_config.hidden_size,
                                cfg.hf_config.vocab_size, len(selected))
        self.weight_report = self.model.load_checkpoint(spec.draft_model,
            runner.model.model.embed_tokens, runner.model.lm_head)
        self._initialize_cache(self.model.config.hidden_size)
        self.last_confidences = {}

    def release(self, slot):
        super().release(slot)
        if hasattr(self, 'last_confidences'):
            self.last_confidences.pop(slot, None)

    def _context(self, slot, end):
        start = self.validated[slot]
        if start > end:
            raise RuntimeError('Block draft cache is ahead of committed target')
        positions = torch.arange(start, end, device=self.features.device)
        kvs = []
        for layer in self.model.layers:
            attention = layer.self_attn
            storage = attention.attn
            shape = (len(self.validated), -1, attention.num_kv_heads, attention.head_dim)
            k, v = storage.k_cache.view(shape)[slot], storage.v_cache.view(shape)[slot]
            if start < end:
                new_k, new_v = attention.context_kv(self.features[slot, start:end], positions)
                k[start:end], v[start:end] = new_k, new_v
            kvs.append((k[:end], v[:end]))
        self.validated[slot] = end
        return kvs

    @torch.inference_mode()
    def propose_device(self, contexts):
        rows, offsets, requests = [], [0], []
        runner = self.runner
        device = self.features.device
        for ctx in contexts:
            slot = self._claim_slot(ctx.request_id)
            end = ctx.computed_length
            capacity = self.model.block_size-(not self.model.sample_from_anchor)
            count = self._draft_count(ctx, capacity=capacity)
            row = torch.empty(0, device=device, dtype=torch.int64)
            if count:
                kvs = self._context(slot, end)
                width = count + (not self.model.sample_from_anchor)
                ids = torch.full((width,), self.model.mask_token_id, device=device, dtype=torch.int64)
                anchor = runner.request_state.tokens.tensor[slot, end]
                ids[0] = anchor
                positions = torch.arange(end, end+width, device=device)
                hidden = self.model(ids, positions, kvs)
                row, confidence = self.model.candidates(hidden, anchor, count)
                self.last_confidences[slot] = confidence
            rows.append(row)
            requests.append(ctx.request_id)
            offsets.append(offsets[-1]+row.numel())
        tokens = torch.cat(rows) if rows else torch.empty(0, device=device, dtype=torch.int64)
        return DeviceDraftProposal(tuple(requests), tokens, tuple(offsets), None)
