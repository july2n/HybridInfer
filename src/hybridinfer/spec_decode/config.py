from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    enabled: bool = False
    method: str = "ngram"
    max_draft_tokens: int = 4
    ngram_min: int = 2
    ngram_max: int = 8
    verification_mode: str = "sequential"

    def __post_init__(self):
        if self.method != "ngram":
            raise ValueError("Only ngram drafts are implemented")
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be positive")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("invalid ngram range")
        if self.verification_mode != "sequential":
            raise ValueError("Only sequential reference verification is implemented")
