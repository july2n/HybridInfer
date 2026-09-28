"""Real-model ragged verification, endpoint and same-path recovery checks.

Diagnostics are intrusive and are not a performance benchmark. Natural n-gram
proposals are used; no baseline output is fed into the proposer.
"""
import argparse
import json
from pathlib import Path

import torch

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.spec_decode import batch_execution
from spec_workloads import natural_cases
from spec_validation import snapshot, equal_state, check_original_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--state-snapshot-budget-mb', type=int, default=256)
    parser.add_argument('--output-tokens', type=int, default=48)
    parser.add_argument('--modes', nargs='+', default=['packed', 'packed_guarded'],
                        choices=['sequential', 'packed_guarded', 'packed'])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--decode-graphs', action='store_true')
    parser.add_argument('--long-prefix', action='store_true')
    parser.add_argument('--json-out', default='logs/validate/spec_batch.json')
    args = parser.parse_args()
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(acceptance_scope='protocol_and_original_endpoint_selection', completed=False, passed=False, scenarios=[], endpoints=[], errors=[])
    engine = None
    try:
        engine = LLMEngine(args.model, max_num_seqs=max(args.batch_sizes), max_model_len=1024,
                           max_num_batched_tokens=256, gpu_memory_utilization=0.6,
                           enforce_eager=not args.decode_graphs, enable_prefix_cache=True,
                           speculative=SpeculativeConfig(enabled=True, state_snapshot_budget_mb=args.state_snapshot_budget_mb))
        runner = engine.model_runner
        original = runner.verify_speculative_batch
        original_remove = runner.remove_request
        final_states = {}
        def remove(seq_id):
            slot = runner.input_batch.seq_id_to_slot.get(seq_id)
            if slot is not None:
                # Speculative finish may have cleared the CPU page table;
                # the resident GPU table still owns the completed mapping.
                c = int(runner.request_state.computed.tensor[slot])
                ids = runner.request_state.block_tables.tensor[slot, :(c+runner.block_size-1)//runner.block_size].long()
                kv = runner.kv_cache.index_select(2, ids)
                kv = kv.reshape(*kv.shape[:2], -1, *kv.shape[-2:])[:, :, :c].cpu().clone()
                final_states[seq_id] = dict(computed=c,
                    tokens=runner.request_state.tokens.tensor[slot, :c+1].cpu().clone(), kv=kv,
                    conv=[layer.conv_states[slot].cpu().clone() for layer in runner.gdn_layers],
                    recurrent=[layer.recurrent_states[slot].cpu().clone() for layer in runner.gdn_layers])
            return original_remove(seq_id)
        runner.remove_request = remove
        active_mode = None
        original_forward = batch_execution.packed_batch_forward
        trial = {}
        def capture_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            trial['endpoints'] = runner._trial_endpoints
            return result
        batch_execution.packed_batch_forward = capture_forward
        def verify(seqs, plans):
            trial.clear()
            output = original(seqs, plans)
            captured_endpoints = trial.get('endpoints')
            get_output = output.get_output
            checked = False
            def consume():
                nonlocal checked
                results = get_output()
                if checked:
                    return results
                checked = True
                actual = [snapshot(runner, s) for s in seqs]
                failures = []
                for s, p, r, state in zip(seqs, plans, results, actual):
                    if (state['computed'] != r.committed_computed_length
                            or state['computed'] != p.computed_length+r.output_length
                            or state['tokens'].tolist() != s.token_ids+list(r.token_ids)):
                        failures.append(f'{s.seq_id}:history/endpoint')
                if active_mode == 'packed' and captured_endpoints:
                    failures.extend(check_original_endpoints(runner, seqs, plans, results, captured_endpoints))
                for state in actual:
                    if not all(torch.isfinite(t).all() for t in
                               [state['kv'], *state['conv'], *state['recurrent']]):
                        failures.append('nonfinite_state')
                record['endpoints'].append(dict(mode=active_mode, requests=len(seqs),
                                                 draft_counts=[len(p.candidates) for p in plans],
                                                 output_lengths=[r.output_length for r in results],
                                                 accepted=[r.accepted_draft_tokens for r in results],
                                                 failures=failures))
                if failures:
                    raise AssertionError(failures)
                return results
            output.get_output = consume
            return output
        runner.verify_speculative_batch = verify
        cases = natural_cases()
        prompts = [engine.tokenizer.encode(text)[:256] for text in
                   [cases['natural_en'], cases['natural_zh'],
                    'Continue: alpha beta alpha beta alpha beta alpha beta.\n', cases['low_match']]]
        if args.long_prefix:
            prompts.insert(0, engine.tokenizer.encode(cases['long_records'])[:509])
        checkpoints = engine.scheduler.checkpoints
        for prefix in (False, True):
            engine.scheduler.enable_prefix_cache = prefix
            engine.scheduler.checkpoints = checkpoints if prefix else None
            for bs in args.batch_sizes:
                selected = [prompts[i % len(prompts)] for i in range(bs)]
                params = [SamplingParams(temperature=0, max_tokens=args.output_tokens-(i % 3), ignore_eos=True)
                          for i in range(bs)]
                engine.config.speculative = engine.scheduler.speculative = None
                final_states = {}
                cold_baseline = engine.generate(selected, params, use_tqdm=False)
                cold_states = [state for _, state in sorted(final_states.items())]
                # Warm prefix caching changes prefill shapes/cohorts even in
                # ordinary decoding. Compare both paths at the same warm-cache
                # condition, and preserve cold/warm baseline drift separately.
                final_states = {}
                baseline_hits = checkpoints.hits if checkpoints is not None else 0
                baseline = engine.generate(selected, params, use_tqdm=False)
                baseline_states = [state for _, state in sorted(final_states.items())]
                baseline_hits = checkpoints.hits-baseline_hits if checkpoints is not None else 0
                cold_warm_matches = [a['token_ids']==b['token_ids'] for a, b in zip(cold_baseline, baseline)]
                cold_warm_states = [equal_state(a, b) for a, b in zip(cold_states, baseline_states)]
                for mode in args.modes:
                    active_mode = mode
                    config = SpeculativeConfig(enabled=True, verification_mode=mode, state_snapshot_budget_mb=args.state_snapshot_budget_mb)
                    engine.config.speculative = engine.scheduler.speculative = config
                    start = len(record['endpoints'])
                    hits = checkpoints.hits if checkpoints is not None else 0
                    metrics_before = dict(runner.spec_metrics)
                    final_states = {}
                    actual = engine.generate(selected, params, use_tqdm=False)
                    actual_states = [state for _, state in sorted(final_states.items())]
                    state_failures = [equal_state(base, trial)
                                      for base, trial in zip(baseline_states, actual_states)]
                    if len(actual_states) != len(selected) or len(baseline_states) != len(selected):
                        raise AssertionError('missing final request state')
                    matches = [a['token_ids'] == b['token_ids'] for a, b in zip(actual, baseline)]
                    endpoints = record['endpoints'][start:]
                    scenario = dict(mode=mode, prefix=prefix, batch_size=bs,
                                    token_matches=matches, baseline_state_differences=state_failures,
                                    baseline_prefix_hits=baseline_hits,
                                    baseline_cold_warm_matches=cold_warm_matches,
                                    baseline_cold_warm_state_differences=cold_warm_states,
                                    batch_rounds=len(endpoints),
                                    recovery_passed=all(not e['failures'] for e in endpoints),
                                    prefix_hits=(checkpoints.hits-hits if checkpoints is not None else 0),
                                    metrics={k: v-metrics_before.get(k, 0) for k, v in runner.spec_metrics.items()})
                    record['scenarios'].append(scenario)
                    print(json.dumps(scenario, ensure_ascii=False), flush=True)
                    path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
                    if not endpoints:
                        record['errors'].append('no batched speculative verification exercised')
        record['completed'] = True
        record['passed'] = not record['errors']
    except Exception as exc:
        record['errors'].append(repr(exc))
        raise
    finally:
        if engine is not None and 'original_forward' in locals():
            batch_execution.packed_batch_forward = original_forward
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        if engine is not None:
            engine.exit()
    print(json.dumps(dict(completed=record['completed'], passed=record['passed'],
                          scenarios=len(record['scenarios']), endpoints=len(record['endpoints'])), indent=2))
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
