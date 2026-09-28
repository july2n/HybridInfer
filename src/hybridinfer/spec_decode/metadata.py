"""Two coordinate systems for ragged, linear target verification."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class VerificationBatch:
    plans: tuple
    scheduled_counts: tuple[int, ...]

    def __post_init__(self):
        if not self.plans or len(self.plans) != len(self.scheduled_counts):
            raise ValueError("verification requires a nonempty aligned batch")
        if len({p.request_id for p in self.plans}) != len(self.plans):
            raise ValueError("duplicate verification request")
        if any(p.computed_length < 0 or n < len(p.input_tokens)
               for p, n in zip(self.plans, self.scheduled_counts)):
            raise ValueError("scheduled inputs must include the anchor and candidates")

    @classmethod
    def from_plans(cls, plans):
        plans = tuple(plans)
        return cls(plans, tuple(len(p.input_tokens) for p in plans))

    @property
    def draft_counts(self):
        return tuple(len(p.candidates) for p in self.plans)

    @property
    def max_draft_tokens(self):
        return max(self.draft_counts)

    def tensors(self, device):
        hidden_indices, target_indices, bonus_indices = [], [], []
        draft_ends, sample_ends, candidates = [], [], []
        input_end = sample_end = draft_end = 0
        for plan, scheduled in zip(self.plans, self.scheduled_counts):
            k = len(plan.candidates)
            input_end += scheduled
            hidden_indices.extend(range(input_end-k-1, input_end))
            target_indices.extend(range(sample_end, sample_end+k))
            bonus_indices.append(sample_end+k)
            candidates.extend(plan.candidates)
            draft_end += k
            sample_end += k+1
            draft_ends.append(draft_end)
            sample_ends.append(sample_end)
        make = lambda values: torch.tensor(values, dtype=torch.int64, device=device)
        return VerificationMetadata(
            make(hidden_indices), make(target_indices), make(bonus_indices),
            make(draft_ends), make(sample_ends), make(candidates),
            make(self.draft_counts), self.max_draft_tokens,
        )


@dataclass(frozen=True, slots=True)
class VerificationMetadata:
    logits_indices: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    draft_token_ids: torch.Tensor
    num_draft_tokens: torch.Tensor
    max_draft_tokens: int

    def select_logits(self, hidden, project):
        """Only this index addresses full hidden states; others address logits."""
        return project(hidden.index_select(0, self.logits_indices))
