from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    seed: int | None = None

    def __post_init__(self):
        assert self.temperature >= 0 and self.temperature < float("inf")
        assert self.max_tokens > 0
        assert self.seed is None or 0 <= self.seed < 2**63
