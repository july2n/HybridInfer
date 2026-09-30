from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    enabled: bool = False
    method: str = "ngram"
    max_draft_tokens: int = 4
    state_snapshot_budget_mb: int = 256
    ngram_min: int = 2
    ngram_max: int = 8
    # Conservative anchor fallback remains the enabled default while native
    # numerical/quality budgets and performance coverage are being expanded.
    verification_mode: str = "packed_guarded"
    mtp_draft_sampling: str = "greedy"
    draft_model: str | None = None
    eagle3_feature_layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.method not in ("ngram", "mtp", "eagle3", "dflash", "dspark"):
            raise ValueError("Implemented draft backends: ngram, mtp, eagle3, dflash, dspark")
        if self.method in ("eagle3", "dflash", "dspark") and not self.draft_model:
            raise ValueError("This backend requires a trained draft_model checkpoint directory")
        if self.eagle3_feature_layers is not None and (not self.eagle3_feature_layers
                or any(type(i) is not int or i < 0 for i in self.eagle3_feature_layers)
                or len(set(self.eagle3_feature_layers)) != len(self.eagle3_feature_layers)):
            raise ValueError("EAGLE-3 feature layers must be distinct nonnegative boundary indices")
        if self.state_snapshot_budget_mb < 1:
            raise ValueError("state_snapshot_budget_mb must be positive")
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be positive")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("invalid ngram range")
        if self.verification_mode not in ("packed", "sequential", "packed_guarded"):
            raise ValueError("verification_mode must be packed, sequential or packed_guarded")
        if self.mtp_draft_sampling not in ("greedy", "random"):
            raise ValueError("mtp_draft_sampling must be greedy or random")
        if self.mtp_draft_sampling == "random" and (self.method != "mtp" or self.verification_mode != "packed"):
            raise ValueError("Random MTP drafts require method=mtp and verification_mode=packed")
