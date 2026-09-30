"""Qwen3.5 MTP draft proposer."""
from hybridinfer.models.qwen3_5_mtp import Qwen3_5MTP
from .interfaces import BackendCapabilities
from .llm_base_proposer import SpecDecodeBaseProposer


class MTPProposer(SpecDecodeBaseProposer):
    capabilities = BackendCapabilities(requires_target_features=True,
                                       supports_random_sampling=True)

    def __init__(self, runner):
        super().__init__(runner)
        cfg = runner.config
        self.model = Qwen3_5MTP(cfg.hf_config)
        self.weight_report = self.model.load_checkpoint(cfg.model)
        self._initialize_cache(cfg.hf_config.hidden_size)
