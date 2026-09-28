"""Ragged linear verification: one target forward, GPU accept and commit.

Native recovery selects original trial endpoints on the GPU. Conservative
modes fall back to ordinary batched anchors to preserve numeric batch shape.
Output D2H and the final commit fence have independent asynchronous ownership.
"""
from contextlib import ExitStack
from time import perf_counter

import torch

from hybridinfer.utils.context import BatchDescriptor, set_context, reset_context
from .async_output import AsyncVerificationOutput
from .batch_verifier import accept_greedy_batch
from .rejection import accept_random_batch
from .commit import commit_batch
from .interfaces import VerificationPlan
from .metadata import VerificationBatch
from .state import GDNTransaction
from .endpoints import begin_endpoints, select_endpoints


@torch.inference_mode()
def packed_batch_forward(runner, seqs, batch, state_slots, *, project=True):
    device = runner.request_state.tokens.tensor.device
    slots = runner.input_batch.slots_for(seqs)
    inputs, positions, mapping, slices, chunks = [], [], [], [], []
    cu_q, cu_k = [0], [0]
    for row, (seq, plan) in enumerate(zip(seqs, batch.plans)):
        n = len(plan.input_tokens)
        slices.append((len(inputs), len(inputs)+n))
        chunks.extend((row, i) for i in range((n+63)//64))
        inputs.extend(plan.input_tokens)
        positions.extend(range(plan.computed_length, plan.trial_end))
        mapping.extend(seq.block_table[p//runner.block_size]*runner.block_size+p%runner.block_size
                       for p in range(plan.computed_length, plan.trial_end))
        cu_q.append(cu_q[-1]+n)
        cu_k.append(cu_k[-1]+plan.trial_end)
    counts = [len(p.input_tokens) for p in batch.plans]
    make = lambda values, dtype=torch.int64: torch.tensor(values, dtype=dtype, device=device)
    set_context(True, cu_seqlens_q=make(cu_q, torch.int32),
                cu_seqlens_k=make(cu_k, torch.int32),
                max_seqlen_q=max(counts), max_seqlen_k=max(p.trial_end for p in batch.plans),
                slot_mapping=make(mapping, torch.int32),
                block_tables=runner.request_state.block_tables.tensor.index_select(0, make(slots)),
                state_indices=make(state_slots), prefill_slices=slices,
                prefill_chunk_indices=make(chunks, torch.int32),
                batch_descriptor=BatchDescriptor('spec_decode', len(inputs), len(seqs),
                                                 counts[0] if len(set(counts)) == 1 else None,
                                                 max(counts)))
    endpoints = begin_endpoints(runner, len(inputs), len(seqs))
    hidden = runner.model(make(inputs), make(positions))
    runner._trial_endpoints = endpoints
    proposer = getattr(runner, 'draft_proposer', None)
    if proposer is not None:
        proposer.record(slots, make(positions), hidden, slices)
    if not project:
        return None
    # Do not reuse ordinary prefill's final-row-only projection.
    return batch.tensors(device).select_logits(hidden, runner.model.compute_logits)


def _decode_batch_anchors(runner, seqs, plans, state_slots):
    """Match ordinary decode's batch shape, advancing each request once."""
    device = runner.request_state.tokens.tensor.device
    slots = runner.input_batch.slots_for(seqs)
    make = lambda values, dtype=torch.int64: torch.tensor(values, dtype=dtype, device=device)
    mapping = [seq.block_table[p.computed_length//runner.block_size]*runner.block_size
               +p.computed_length%runner.block_size for seq, p in zip(seqs, plans)]
    set_context(False, slot_mapping=make(mapping, torch.int32),
                context_lens=make([p.computed_length+1 for p in plans], torch.int32),
                block_tables=runner.request_state.block_tables.tensor.index_select(0, make(slots)),
                state_indices=make(state_slots),
                batch_descriptor=BatchDescriptor('spec_decode', len(seqs), len(seqs), 1, 1))
    hidden = runner.model(make([p.anchor for p in plans]), make([p.computed_length for p in plans]))
    proposer = getattr(runner, 'draft_proposer', None)
    if proposer is not None:
        proposer.record(make(slots), make([p.computed_length for p in plans]), hidden)
    runner._anchor_logits = runner.model.compute_logits(hidden)
    return runner._anchor_logits.argmax(-1)


def _trial_kv(runner, seq, plan):
    cache = getattr(runner, 'kv_cache', None)
    if cache is None:
        return None
    mapping = [seq.block_table[p//runner.block_size]*runner.block_size+p%runner.block_size
               for p in range(plan.computed_length, plan.trial_end)]
    flat = cache.reshape(*cache.shape[:2], -1, *cache.shape[-2:])
    return flat.index_select(2, torch.tensor(mapping, device=cache.device))


@torch.inference_mode()
def verify_speculative_batch(runner, seqs, plans):
    seqs, plans = tuple(seqs), tuple(plans)
    batch = VerificationBatch.from_plans(plans)
    if len(seqs) != len(plans) or len(seqs) > runner.config.max_num_seqs:
        raise ValueError('verification batch size disagrees with request capacity')
    if runner._pending is not None or runner.world_size != 1:
        raise RuntimeError('verification requires an idle single-GPU runner')
    for seq, plan in zip(seqs, plans):
        if seq.temperature != 0 and runner.config.speculative.verification_mode != 'packed':
            raise ValueError('Random speculative sampling requires packed verification')
        if (plan.request_id != seq.seq_id or plan.computed_length != seq.num_cached_tokens
                or seq.num_tokens != plan.computed_length+1 or plan.anchor != seq.last_token
                or plan.trial_end > runner.config.max_model_len):
            raise ValueError('verification plan disagrees with committed sequence/context')
        if seq.seq_id not in runner.input_batch.seq_id_to_slot:
            raise RuntimeError('verification requires resident prefill state')
    runner.input_batch.update(seqs)
    slots = runner.input_batch.slots_for(seqs)
    runner.request_state.update(seqs, slots, [])
    state = runner.request_state
    device = state.tokens.tensor.device
    slots_t = torch.tensor(slots, device=device)
    expected = torch.tensor([p.computed_length for p in plans], device=device,
                            dtype=state.computed.tensor.dtype)
    if not torch.equal(state.computed.tensor.index_select(0, slots_t), expected):
        raise RuntimeError('CPU/GPU committed lengths disagree')
    private = [runner.config.max_num_seqs+i for i in range(len(seqs))]
    if any(pool.shape[0] <= private[-1] for layer in runner.gdn_layers
           for pool in (layer.conv_states, layer.recurrent_states)):
        raise RuntimeError('private GDN verification slot capacity exhausted')
    saved_tokens = [state.tokens.tensor[slot, p.computed_length+1:p.trial_end+1].clone()
                    for slot, p in zip(slots, plans)]
    saved_last = runner.sampled_token_ids_gpu.index_select(0, slots_t).clone()
    mode = runner.config.speculative.verification_mode
    guarded, used_packed = mode == 'packed_guarded', mode == 'packed'
    stats = runner.spec_metrics
    extra = dict(batch_rounds=1, batch_requests=len(seqs),
                 batch_zero_draft_requests=sum(k == 0 for k in batch.draft_counts),
                 planned_trial_tokens=sum(batch.scheduled_counts))

    def clock():
        if device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return perf_counter()

    def elapsed(a, b):
        return a.elapsed_time(b)/1000 if device.type == 'cuda' else b-a

    try:
        start = clock()
        with ExitStack() as stack:
            transactions = [stack.enter_context(GDNTransaction(runner.gdn_layers, src, dst))
                            for src, dst in zip(slots, private)]
            copied = clock()
            native = None
            saved_states = saved_kv = None
            resource_failure = False
            if mode != 'sequential':
                try:
                    native_logits = packed_batch_forward(runner, seqs, batch, private)
                    native = native_logits.argmax(-1)
                    if guarded:
                        saved_states = [None if p.candidates else
                                        [pool[dst].clone() for pool, _ in txn.original]
                                        for p, txn, dst in zip(plans, transactions, private)]
                        saved_kv = [None if p.candidates else _trial_kv(runner, seq, p)
                                    for seq, p in zip(seqs, plans)]
                except torch.cuda.OutOfMemoryError:
                    native = None
                    resource_failure = True
                    used_packed = False
                if guarded or resource_failure:
                    for txn, dst in zip(transactions, private):
                        for pool, original in txn.original:
                            pool[dst].copy_(original)
            packed_end = clock()
            anchor_only = not used_packed
            if used_packed:
                predictions = native
            else:
                # A per-request batch=1 reference changes GEMM shape and can
                # perturb subsequent greedy winners. Preserve ordinary batch
                # shape and only commit these reference anchor endpoints.
                predictions = _decode_batch_anchors(runner, seqs, plans, private)
                extra['batch_reference_anchor_only'] = len(seqs)
            verified = clock()
            if mode != 'sequential':
                extra.update(packed_rounds=len(seqs), packed_trial_tokens=sum(batch.scheduled_counts),
                             packed_resource_fallbacks=len(seqs) if resource_failure else 0,
                             reference_trial_tokens=0 if used_packed else len(seqs))
                fallbacks = 0
                offset = 0
                for row, (seq, plan, txn, dst) in enumerate(zip(seqs, plans, transactions, private)):
                    n = len(plan.input_tokens)
                    token_diff = state_diff = kv_diff = False
                    if guarded and not resource_failure:
                        token_diff = not torch.equal(native[offset:offset+1], predictions[row:row+1])
                        # Native final state is at a different endpoint when
                        # K>0; comparing it with anchor state would be invalid.
                        if not plan.candidates:
                            state_diff = any(not torch.equal(saved, pool[dst])
                                             for saved, (pool, _) in zip(saved_states[row], txn.original))
                            kv_diff = saved_kv[row] is not None and not torch.equal(saved_kv[row], _trial_kv(runner, seq, plan))
                        else:
                            extra['packed_state_unchecked_requests'] = extra.get('packed_state_unchecked_requests', 0)+1
                    fallbacks += int(resource_failure or (guarded and bool(plan.candidates))
                                     or token_diff or state_diff or kv_diff)
                    if guarded:
                        for key, value in (('packed_token_mismatches', token_diff),
                                           ('packed_state_mismatches', state_diff), ('packed_kv_mismatches', kv_diff)):
                            extra[key] = extra.get(key, 0)+int(value)
                    offset += n
                extra['packed_fallbacks'] = fallbacks
            acceptance_batch = VerificationBatch.from_plans(
                [VerificationPlan(p.request_id, p.computed_length, p.anchor, ()) for p in plans]
            ) if anchor_only else batch
            if any(s.temperature != 0 for s in seqs) and used_packed:
                draft_probs = None
                if any(p.draft_probabilities is not None for p in plans):
                    probs = []
                    for p in plans:
                        q = p.draft_probabilities
                        if q is None:
                            q = torch.zeros((len(p.candidates), native_logits.shape[1]), device=device)
                            if p.candidates:
                                q.scatter_(1, torch.tensor(p.candidates, device=device)[:, None], 1.)
                        probs.append(q)
                    draft_probs = torch.cat(probs)
                acceptance = accept_random_batch(batch, batch.tensors(device), native_logits,
                    draft_probs=draft_probs,
                    temperatures=[s.temperature for s in seqs],
                    seeds=[s.seed if s.seed is not None else torch.initial_seed()+s.seq_id for s in seqs],
                    remaining_output_tokens=[s.max_tokens-s.num_completion_tokens for s in seqs],
                    max_model_len=runner.config.max_model_len, eos=runner.config.eos,
                    ignore_eos=[s.ignore_eos for s in seqs])
            elif any(s.temperature != 0 for s in seqs):
                # Resource fallback is an ordinary sampled anchor, not argmax.
                hidden_logits = runner._anchor_logits
                sampled = runner.sampler.sample(hidden_logits, slots_t,
                    state.temperatures.tensor, state.seeds.tensor,
                    state.computed.tensor+1)
                acceptance = accept_greedy_batch(acceptance_batch, acceptance_batch.tensors(device), sampled,
                    remaining_output_tokens=[s.max_tokens-s.num_completion_tokens for s in seqs],
                    max_model_len=runner.config.max_model_len, eos=runner.config.eos,
                    ignore_eos=[s.ignore_eos for s in seqs])
            else:
                acceptance = accept_greedy_batch(
                    acceptance_batch, acceptance_batch.tensors(device), predictions,
                    remaining_output_tokens=[s.max_tokens-s.num_completion_tokens for s in seqs],
                    max_model_len=runner.config.max_model_len, eos=runner.config.eos,
                    ignore_eos=[s.ignore_eos for s in seqs])
            if anchor_only:
                for txn in transactions:
                    txn.commit_trial()
            else:
                endpoints = getattr(runner, '_trial_endpoints', {})
                if set(endpoints) != {layer.layer_idx for layer in runner.gdn_layers}:
                    raise RuntimeError('native verification is missing original GDN endpoints')
                starts = torch.tensor([0, *batch.scheduled_counts[:-1]], device=device).cumsum(0)
                select_endpoints(endpoints, slots_t, starts+acceptance.lengths-1)
                for txn in transactions:
                    txn.finish_endpoint_commit()
            committed = clock()
            commit_batch(state, runner.sampled_token_ids_gpu, slots_t, acceptance)
            done = clock()

            def on_ready(results):
                if anchor_only:
                    extra['reference_first_draft_matches'] = sum(bool(p.candidates) and r.token_ids[0] == p.candidates[0]
                                                                  for p, r in zip(plans, results))
                for key, value in extra.items():
                    stats[key] = stats.get(key, 0)+value
                for key, value in (
                    ('rounds', len(seqs)), ('draft_tokens', sum(batch.draft_counts)),
                    ('accepted_tokens', sum(r.accepted_draft_tokens for r in results)),
                    ('output_tokens', sum(r.output_length for r in results)),
                    ('trial_tokens', sum(r.trial_computed_tokens for r in results)),
                    ('replay_tokens', 0),
                    ('copy_seconds', elapsed(start, copied)), ('verify_seconds', elapsed(copied, verified)),
                    ('restore_seconds', elapsed(verified, committed)), ('commit_seconds', elapsed(committed, done))):
                    stats[key] = stats.get(key, 0)+value
                if mode != 'sequential':
                    stats['packed_seconds'] = stats.get('packed_seconds', 0)+elapsed(copied, packed_end)
                    stats['reference_seconds'] = stats.get('reference_seconds', 0)+elapsed(packed_end, verified)
            result_plans = plans if native is not None else acceptance_batch.plans
            output = AsyncVerificationOutput(acceptance, result_plans, runner, on_ready)
        return output
    except Exception:
        for slot, plan, original in zip(slots, plans, saved_tokens):
            state.tokens.tensor[slot, plan.computed_length+1:plan.trial_end+1].copy_(original)
        state.computed.tensor[slots_t] = expected
        runner.sampled_token_ids_gpu[slots_t] = saved_last
        # The scheduler may release trial pages after this exception returns.
        if device.type == 'cuda':
            torch.cuda.current_stream().synchronize()
        raise
    finally:
        runner._trial_endpoints = {}
        reset_context()
