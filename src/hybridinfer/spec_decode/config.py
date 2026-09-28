from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeculativeConfig:
    enabled: bool = False
    method: str = "ngram"
    max_draft_tokens: int = 4
    ngram_min: int = 2
    ngram_max: int = 8
    # Native BF16 packed verification can change greedy winners and future
    # decode state. Keep exact reference fallback as the enabled default until
    # the natural-input token acceptance gate passes; native stays explicit.
    verification_mode: str = "packed_guarded"

    def __post_init__(self):
        if self.method != "ngram":
            raise ValueError("Only ngram drafts are implemented")
        if self.max_draft_tokens < 1:
            raise ValueError("max_draft_tokens must be positive")
        if not 1 <= self.ngram_min <= self.ngram_max:
            raise ValueError("invalid ngram range")
        if self.verification_mode not in ("packed", "sequential", "packed_guarded"):
            raise ValueError("verification_mode must be packed, sequential or packed_guarded")
