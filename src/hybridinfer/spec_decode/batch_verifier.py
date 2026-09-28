"""GPU greedy longest-prefix acceptance with explicit output endpoints."""
from dataclasses import dataclass

import torch

from .interfaces import VerificationResult

# Match the CPU reference's precedence: EOS, output limit, context limit.
FINISH_REASONS = (None, 'eos', 'max_tokens', 'context')


@dataclass(frozen=True, slots=True)
class BatchAcceptance:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    accepted: torch.Tensor
    computed: torch.Tensor
    reasons: torch.Tensor

    def payload(self):
        return torch.cat((self.token_ids, self.lengths[:, None], self.accepted[:, None],
                          self.computed[:, None], self.reasons[:, None]), dim=1)


def results_from_payload(payload, plans):
    rows = payload.tolist()
    results = []
    for row, plan in zip(rows, plans):
        length, accepted, computed, reason = row[-4:]
        results.append(VerificationResult(tuple(row[:length]), accepted,
                                          len(plan.input_tokens), computed,
                                          bool(reason), FINISH_REASONS[reason]))
    return results


@torch.inference_mode()
def accept_greedy_batch(batch, metadata, predictions, *, remaining_output_tokens,
                        max_model_len, eos=-1, ignore_eos=None):
    """predictions is in selected-logits coordinates, containing K_i+1 rows."""
    if predictions.ndim != 1 or predictions.numel() != sum(k+1 for k in batch.draft_counts):
        raise ValueError('verification requires exactly K_i+1 predictions per request')
    remaining = tuple(remaining_output_tokens)
    ignored = tuple(ignore_eos) if ignore_eos is not None else (False,) * len(batch.plans)
    if len(remaining) != len(batch.plans) or len(ignored) != len(batch.plans):
        raise ValueError('sampling budgets must match the batch')
    if any(r < 1 or p.trial_end > max_model_len or p.computed_length >= max_model_len
           for p, r in zip(batch.plans, remaining)):
        raise ValueError('no output/context budget or trial exceeds context')
    device = predictions.device
    width = metadata.max_draft_tokens+1
    pos = torch.arange(width, device=device)[None, :]
    starts = metadata.cu_num_sampled_tokens - metadata.num_draft_tokens - 1
    indices = starts[:, None] + pos
    valid_rows = pos <= metadata.num_draft_tokens[:, None]
    padded = predictions[indices.clamp(max=predictions.numel()-1)]
    draft_starts = metadata.cu_num_draft_tokens - metadata.num_draft_tokens
    if metadata.draft_token_ids.numel():
        draft = metadata.draft_token_ids[
            (draft_starts[:, None]+pos).clamp(max=metadata.draft_token_ids.numel()-1)]
        matches = (pos < metadata.num_draft_tokens[:, None]) & (padded == draft)
        accepted_prefix = matches.to(torch.int64).cumprod(1).sum(1)
    else:
        accepted_prefix = torch.zeros(len(batch.plans), dtype=torch.int64, device=device)
    computed = torch.tensor([p.computed_length for p in batch.plans], device=device)
    remaining_t = torch.tensor(remaining, device=device)
    lengths = torch.minimum(accepted_prefix+1, torch.minimum(remaining_t, max_model_len-computed))
    eos_mask = ((padded == eos) & (pos < lengths[:, None]) & valid_rows
                & ~torch.tensor(ignored, device=device)[:, None])
    first_eos = torch.where(eos_mask, pos+1, width+1).amin(1)
    eos_stop = first_eos <= lengths
    lengths = torch.minimum(lengths, first_eos)
    reasons = torch.where(eos_stop, 1, torch.where(lengths >= remaining_t, 2,
                          torch.where(computed+lengths >= max_model_len, 3, 0)))
    output = torch.where(pos < lengths[:, None], padded, -1)
    return BatchAcceptance(output, lengths, torch.minimum(accepted_prefix, lengths),
                           computed+lengths, reasons)
