import torch
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    topology: str = "linear"
    requires_target_features: bool = False
    supports_random_sampling: bool = False


@dataclass(frozen=True, slots=True)
class DraftContext:
    request_id: int
    token_ids: tuple[int, ...]
    computed_length: int
    remaining_output_tokens: int
    max_model_len: int
    verification_budget: int
    temperature: float = 0.0
    seed: int | None = None

    def __post_init__(self):
        if len(self.token_ids) != self.computed_length + 1:
            raise ValueError("decode history must have exactly one uncomputed anchor")
        if min(self.computed_length, self.remaining_output_tokens,
               self.max_model_len, self.verification_budget) < 0:
            raise ValueError("lengths and budgets must be nonnegative")
        if not 0 <= self.temperature < float("inf"):
            raise ValueError("draft temperature must be finite and nonnegative")
        if self.seed is not None and not 0 <= self.seed < 2**63:
            raise ValueError("draft seed must be in [0, 2**63)")


@dataclass(frozen=True, slots=True)
class DraftProposal:
    request_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    offsets: tuple[int, ...]
    probabilities: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("duplicate draft request")
        if (len(self.offsets) != len(self.request_ids) + 1 or not self.offsets
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.token_ids)
                or any(a > b for a, b in zip(self.offsets, self.offsets[1:]))):
            raise ValueError("invalid packed draft offsets")

    def tokens_for(self, row):
        return self.token_ids[self.offsets[row]:self.offsets[row + 1]]


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    request_id: int
    computed_length: int
    anchor: int
    candidates: tuple[int, ...]
    draft_probabilities: torch.Tensor | None = field(default=None, repr=False, compare=False)

    @property
    def input_tokens(self):
        return (self.anchor, *self.candidates)

    @property
    def trial_end(self):
        return self.computed_length + len(self.input_tokens)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    token_ids: tuple[int, ...]
    accepted_draft_tokens: int
    trial_computed_tokens: int
    committed_computed_length: int
    finished: bool
    finish_reason: str | None = None

    @property
    def output_length(self):
        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class DeviceDraftProposal:
    """Packed CUDA candidates; q=None explicitly means point-mass drafts.

    Probabilistic backends supply the probabilities actually used to
    sample the candidates. Offsets include a leading zero (B+1 coordinates).
    """
    request_ids: tuple[int, ...]
    token_ids: torch.Tensor
    offsets: tuple[int, ...]
    probabilities: torch.Tensor | None = None

    def __post_init__(self):
        if self.token_ids.ndim != 1 or self.token_ids.dtype != torch.int64:
            raise ValueError("Device draft tokens must be a flat int64 tensor")
        if (len(set(self.request_ids)) != len(self.request_ids)
                or len(self.offsets) != len(self.request_ids)+1
                or self.offsets[0] != 0 or self.offsets[-1] != self.token_ids.numel()
                or any(a > b for a, b in zip(self.offsets, self.offsets[1:]))):
            raise ValueError("Invalid device draft layout")
        if self.probabilities is not None and (self.probabilities.ndim != 2
                or self.probabilities.shape[0] != self.token_ids.numel()
                or self.probabilities.device != self.token_ids.device):
            raise ValueError("Draft q must align with device tokens")

    def to_host(self):
        return DraftProposal(self.request_ids, tuple(self.token_ids.cpu().tolist()), self.offsets, self.probabilities)
