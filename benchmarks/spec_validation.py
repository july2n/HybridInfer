"""Shared speculative validation metrics; protocol failures and drift stay separate."""
import math
import statistics

import torch


def conv_history_reference(raw, pool, slots, cu):
    """Independent Torch window oracle; no production snapshot kernel."""
    bounds, width = cu.cpu().tolist(), pool.shape[-1]
    windows = []
    for slot, start, end in zip(slots.cpu().tolist(), bounds, bounds[1:]):
        history = torch.cat((pool[slot], raw[start:end].T), dim=-1)
        windows.extend(history[:, i+1:i+1+width] for i in range(end-start))
    return torch.stack(windows)


def snapshot(runner, seq):
    slot = runner.input_batch.slots_for([seq])[0]
    c = int(runner.request_state.computed.tensor[slot])
    pages = (c+runner.block_size-1)//runner.block_size
    ids = runner.request_state.block_tables.tensor[slot, :pages].long()
    kv = runner.kv_cache.index_select(2, ids)
    kv = kv.reshape(*kv.shape[:2], -1, *kv.shape[-2:])[:, :, :c].cpu().clone()
    return dict(computed=c, tokens=runner.request_state.tokens.tensor[slot, :c+1].cpu().clone(),
                kv=kv, conv=[layer.conv_states[slot].cpu().clone() for layer in runner.gdn_layers],
                recurrent=[layer.recurrent_states[slot].cpu().clone() for layer in runner.gdn_layers])


def equal_state(reference, trial):
    """Exact comparison for copies/endpoints; cross-path callers report it only."""
    failures = [key for key in ('tokens', 'kv') if not torch.equal(reference[key], trial[key])]
    for key in ('conv', 'recurrent'):
        if len(reference[key]) != len(trial[key]) or any(
                not torch.equal(a, b) for a, b in zip(reference[key], trial[key])):
            failures.append(key)
    if reference['computed'] != trial['computed']:
        failures.append('computed')
    return failures


def check_original_endpoints(runner, seqs, plans, results, endpoints):
    """Inspect the captured original trial, without target replay or decoding."""
    if len(endpoints) != len(runner.gdn_layers):
        raise AssertionError('original trial is missing GDN endpoints')
    if not len(seqs) == len(plans) == len(results):
        raise AssertionError('request/plan/result counts disagree')
    failures, offset = [], 0
    for seq, plan, result in zip(seqs, plans, results):
        slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
        index = offset+result.output_length-1
        fields = []
        if not 0 < result.output_length <= len(plan.input_tokens):
            raise AssertionError('committed endpoint outside original trial')
        for layer, conv, recurrent in endpoints.values():
            if not torch.equal(layer.conv_states[slot], conv[index]):
                fields.append(f'conv:{layer.layer_idx}')
            if not torch.equal(layer.recurrent_states[slot], recurrent[index]):
                fields.append(f'recurrent:{layer.layer_idx}')
        if fields:
            failures.append(dict(request=seq.seq_id, failed=fields))
        offset += len(plan.input_tokens)
    return failures


def prediction_diagnostics(logits, positions, source):
    top = logits.float().topk(2, dim=-1)
    winners = logits.argmax(-1).tolist()
    positions = list(positions)
    if len(positions) != len(winners):
        raise ValueError('prediction positions must match projected rows')
    return [dict(position=position, token=winner,
                 second_token=ids[1] if ids[0] == winner else ids[0],
                 margin=values[0]-values[1], source=source, committed=False)
            for position, winner, values, ids in
            zip(positions, winners, top.values.tolist(), top.indices.tolist())]


def numerical_summary(rows, *, max_mean_tv=None, max_tv=None, max_flip_rate=None):
    """Optional declared budgets; exact logits are a diagnostic, never a gate."""
    if not rows:
        raise ValueError('numerical validation requires predictions')
    budgets = dict(mean_tv=max_mean_tv, max_tv=max_tv, flip_rate=max_flip_rate)
    if any(value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
           for value in budgets.values()):
        raise ValueError('probability budgets must be finite and between zero and one')
    tv = [row['probability_tv'] for row in rows]
    kl = [row['probability_kl'] for row in rows]
    finite = all(row['logits']['finite'] and math.isfinite(t) and math.isfinite(k)
                 for row, t, k in zip(rows, tv, kl))
    summary = dict(predictions=len(rows), finite=finite,
                   mean_tv=statistics.mean(tv), max_tv=max(tv),
                   mean_kl=statistics.mean(kl),
                   flip_rate=sum(not row['argmax_equal'] for row in rows)/len(rows),
                   exact_logits=sum(row['logits']['equal'] for row in rows), budgets=budgets)
    configured = {key: value for key, value in budgets.items() if value is not None}
    summary['budget_checked'] = bool(configured)
    summary['budget_passed'] = (finite and all(summary[key] <= value for key, value in configured.items())) if configured else None
    return summary
