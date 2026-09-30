"""Shared linear rejection sampler: point-mass or probabilistic draft q.

RNG domains are separate for acceptance, residual correction and bonus. Draft
probabilities must be the realized distribution used to draw each candidate.
"""
import hashlib
import torch
import triton
import triton.language as tl
from .batch_verifier import accept_greedy_batch, finish_batch


def stream_seed(seed, position, domain):
    data = f'{seed}:{position}:{domain}'.encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), 'little') & ((1 << 63)-1)


def random_uniform(seed, position, domain, device, shape=()):
    generator = torch.Generator(device=device)
    generator.manual_seed(stream_seed(seed, position, domain))
    return torch.rand(shape, device=device, generator=generator)


@triton.jit
def _rejection_decisions(P, Q, U, OUT, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    p = tl.load(P+i, i < N, other=0)
    q = tl.load(Q+i, i < N, other=0)
    u = tl.load(U+i, i < N, other=0)
    # Keep vLLM's division order, including its exact threshold behavior.
    accepted = (q > 0) & (p / q >= u)
    tl.store(OUT+i, accepted, i < N)


def rejection_decisions(p_candidate, q_candidate, uniforms):
    if p_candidate.is_cuda:
        p, q, u = (x.contiguous() for x in (p_candidate, q_candidate, uniforms))
        output = torch.empty_like(p, dtype=torch.bool)
        _rejection_decisions[(triton.cdiv(p.numel(), 128),)](p, q, u, output, p.numel(), 128)
        return output
    return (q_candidate > 0) & (p_candidate / q_candidate >= uniforms)


def residual_distribution(p, q):
    residual = (p-q).clamp_min(0)
    return residual / residual.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)


def categorical(probabilities, uniform):
    # searchsorted clamps floating-point cumulative roundoff at the last token.
    return torch.searchsorted(probabilities.cumsum(-1).contiguous(),
                              uniform.reshape(-1).contiguous(), right=True).clamp_max(probabilities.numel()-1).reshape(uniform.shape)


@torch.inference_mode()
def accept_random_batch(batch, metadata, logits, *, temperatures, seeds,
                        draft_probs=None, **limits):
    if len(temperatures) != len(batch.plans) or len(seeds) != len(batch.plans):
        raise ValueError('Sampling settings must match verification requests')
    if logits.ndim != 2 or logits.shape[0] != sum(k+1 for k in batch.draft_counts):
        raise ValueError('Expected K+1 target logits per request')
    if draft_probs is not None and (draft_probs.shape != (sum(batch.draft_counts), logits.shape[1])
            or draft_probs.device != logits.device or not draft_probs.is_floating_point()):
        raise ValueError('Draft probabilities must align with packed candidates, vocabulary and device')
    device = logits.device
    width = metadata.max_draft_tokens+1
    output = torch.full((len(batch.plans), width), -1, dtype=torch.int64, device=device)
    prefixes = []
    sample_start = draft_start = 0
    for row, (plan, temperature, seed) in enumerate(zip(batch.plans, temperatures, seeds)):
        k = len(plan.candidates)
        target = logits[sample_start:sample_start+k+1]
        if temperature == 0:
            winners = target.argmax(-1)
            draft = torch.tensor(plan.candidates, device=device, dtype=torch.int64)
            matches = (winners[:k] == draft).long().cumprod(0)
            prefix = matches.sum()
            output[row, :k+1] = winners
        else:
            p = torch.softmax(target.float()/temperature, -1)
            accepted = []
            for step, token in enumerate(plan.candidates):
                position = plan.computed_length+step+1
                if draft_probs is None:
                    q = torch.zeros_like(p[step])
                    q[token] = 1
                else:
                    q = draft_probs[draft_start+step].float()
                draw = random_uniform(seed, position, 'accept', device)
                decision = rejection_decisions(p[step, token], q[token], draw)
                accepted.append(decision)
                correction = categorical(residual_distribution(p[step], q),
                    random_uniform(seed, position, 'correction', device))
                output[row, step] = torch.where(decision, token, correction)
            prefix = (torch.stack(accepted).long().cumprod(0).sum() if k
                      else torch.zeros((), dtype=torch.int64, device=device))
            output[row, k] = categorical(p[k], random_uniform(seed,
                plan.computed_length+k+1, 'bonus', device))
        prefixes.append(prefix)
        sample_start += k+1
        draft_start += k
    return finish_batch(batch, output, torch.stack(prefixes), **limits)


class RejectionSampler:
    """GPU target acceptance for greedy and probabilistic draft proposals."""

    def sample_greedy(self, batch, metadata, predictions, **limits):
        return accept_greedy_batch(batch, metadata, predictions, **limits)

    def sample_random(self, batch, metadata, logits, **options):
        return accept_random_batch(batch, metadata, logits, **options)
