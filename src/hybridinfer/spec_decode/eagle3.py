"""EAGLE-3 linear proposals using target intermediate features."""
from hybridinfer.models.eagle3 import Eagle3Draft, read_eagle_config
from .interfaces import BackendCapabilities
from .llm_base_proposer import SpecDecodeBaseProposer


class EagleProposer(SpecDecodeBaseProposer):
    capabilities = BackendCapabilities(requires_target_features=True)

    def __init__(self, runner):
        super().__init__(runner, pass_hidden_states_to_model=True)
        spec, cfg = runner.config.speculative, runner.config
        draft = read_eagle_config(spec.draft_model)
        checkpoint_layers = (draft.get('eagle_config') or {}).get('eagle_aux_hidden_state_layer_ids')
        selected = spec.eagle3_feature_layers or checkpoint_layers or (2, cfg.hf_config.num_hidden_layers//2, cfg.hf_config.num_hidden_layers-3)
        self.feature_layers = tuple(selected)
        if (len(set(selected)) != len(selected) or any(type(i) is not int or not 0 <= i < cfg.hf_config.num_hidden_layers for i in selected)):
            raise ValueError('Invalid EAGLE-3 target input-boundary indices')
        if checkpoint_layers is not None and tuple(checkpoint_layers) != self.feature_layers:
            raise ValueError('EAGLE-3 feature selection differs from trained checkpoint')
        self.model = Eagle3Draft(draft, cfg.hf_config.hidden_size, cfg.hf_config.vocab_size, len(selected))
        self.weight_report = self.model.load_checkpoint(spec.draft_model, runner.model.model.embed_tokens)
        self._initialize_cache(self.model.config.hidden_size)

    def _embed_tokens(self, ids):
        return self.model.embed_tokens(ids)

    def _compute_logits(self, hidden):
        return self.model.compute_logits(hidden)


# Keep the public name used by existing callers and validation tools.
Eagle3Proposer = EagleProposer
