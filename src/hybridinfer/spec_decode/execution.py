"""Synchronous single-request reference path; deliberately makes no speed claim."""
from time import perf_counter

import torch

from hybridinfer.utils.context import BatchDescriptor, set_context, reset_context
from .state import GDNTransaction
from .verifier import accept_greedy


@torch.inference_mode()
def verify_sequential(runner, seq, plan):
    if runner._pending is not None or runner.world_size != 1:
        raise RuntimeError('verification requires an idle single-GPU runner')
    if seq.temperature != 0:
        raise ValueError('reference verification requires temperature=0')
    if (plan.request_id != seq.seq_id or plan.computed_length != seq.num_cached_tokens
            or seq.num_tokens != plan.computed_length + 1 or plan.anchor != seq.last_token):
        raise ValueError('verification plan disagrees with committed sequence')
    if plan.trial_end > runner.config.max_model_len:
        raise ValueError('verification trial exceeds context')
    if seq.seq_id not in runner.input_batch.seq_id_to_slot:
        raise RuntimeError('verification requires resident prefill state')
    runner.input_batch.update([seq])
    slot = runner.input_batch.slots_for([seq])[0]
    # A decode request must already have its prefill state. Do not initialize
    # a replacement slot here: that would silently lose the recurrent prefix.
    runner.request_state.update([seq], [slot], [])
    state = runner.request_state
    if int(state.computed.tensor[slot].item()) != plan.computed_length:
        raise RuntimeError('CPU/GPU committed lengths disagree')
    device = state.tokens.tensor.device
    private_slot = runner.config.max_num_seqs
    stats = runner.spec_metrics
    original_tokens = state.tokens.tensor[slot, plan.computed_length+1:plan.trial_end+1].clone()
    original_last = runner.sampled_token_ids_gpu[slot].clone()

    def forward(token, index, state_slot, project=True):
        position = plan.computed_length + index
        page = seq.block_table[position // runner.block_size]
        set_context(False,
                    slot_mapping=torch.tensor([page * runner.block_size + position % runner.block_size],
                                              dtype=torch.int32, device=device),
                    context_lens=torch.tensor([position+1], dtype=torch.int32, device=device),
                    block_tables=state.block_tables.tensor[slot:slot+1],
                    state_indices=torch.tensor([state_slot], dtype=torch.int64, device=device),
                    batch_descriptor=BatchDescriptor(mode='spec_decode', num_tokens=1,
                                                     num_reqs=1, uniform_token_count=1, max_query_len=1))
        hidden = runner.model(torch.tensor([token], dtype=torch.int64, device=device),
                              torch.tensor([position], dtype=torch.int64, device=device))
        return runner.model.compute_logits(hidden) if project else None

    def clock():
        # This is a correctness reference. Synchronize so phase costs include
        # GPU work rather than reporting enqueue time as execution time.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return perf_counter()

    try:
        start = clock()
        with GDNTransaction(runner.gdn_layers, slot, private_slot) as txn:
            copied = clock()
            predictions = [int(forward(token, i, private_slot).argmax(-1).item())
                           for i, token in enumerate(plan.input_tokens)]
            verified = clock()
            result = accept_greedy(plan, predictions,
                                   remaining_output_tokens=seq.max_tokens-seq.num_completion_tokens,
                                   max_model_len=runner.config.max_model_len,
                                   eos=runner.config.eos, ignore_eos=seq.ignore_eos)
            count = result.committed_computed_length - plan.computed_length
            def replay():
                for i, token in enumerate(plan.input_tokens[:count]):
                    forward(token, i, slot, project=False)
            txn.commit(all_inputs_committed=count == len(plan.input_tokens), replay=replay)
            committed = clock()
            # Candidates were never placed in the resident token buffer.
            begin = plan.computed_length + 1
            state.tokens.tensor[slot, begin:begin+result.output_length].copy_(
                torch.tensor(result.token_ids, dtype=torch.int64, device=device))
            state.computed.tensor[slot] = result.committed_computed_length
            runner.sampled_token_ids_gpu[slot] = result.token_ids[-1]
            done = clock()
        stats['rounds'] += 1
        stats['draft_tokens'] += len(plan.candidates)
        stats['accepted_tokens'] += result.accepted_draft_tokens
        stats['output_tokens'] += result.output_length
        stats['trial_tokens'] += len(plan.input_tokens)
        stats['replay_tokens'] += count if count < len(plan.input_tokens) else 0
        stats['copy_seconds'] += copied-start
        stats['verify_seconds'] += verified-copied
        stats['restore_seconds'] += committed-verified
        stats['commit_seconds'] += done-committed
        return result
    except Exception:
        state.tokens.tensor[slot, plan.computed_length+1:plan.trial_end+1].copy_(original_tokens)
        state.computed.tensor[slot] = plan.computed_length
        runner.sampled_token_ids_gpu[slot].copy_(original_last)
        raise
    finally:
        reset_context()
