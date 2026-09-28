"""Natural-input, long-generation and near-tie diagnostics (not a timing benchmark)."""
import argparse
import json
from pathlib import Path

import torch

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.utils.context import get_context, reset_context
from spec_workloads import natural_cases, prepare_prompt
from validate_spec_decode import snapshot, same_block_reference, equal_state, state_drift


def prediction_diagnostics(logits, positions, source):
    top = logits.float().topk(2, dim=-1)
    values, ids = top.values.tolist(), top.indices.tolist()
    winners = logits.argmax(-1).tolist()
    positions = list(positions)
    if len(positions) != len(winners):
        raise ValueError('prediction positions must match projected rows')
    rows = []
    for position, value, tokens, winner in zip(positions, values, ids, winners):
        # topk has a different tie order from argmax. Report the token the
        # verifier actually uses, including when BF16 rounds scores equal.
        second = tokens[1] if tokens[0] == winner else tokens[0]
        rows.append(dict(position=position, token=winner, second_token=second,
                         margin=value[0]-value[1], source=source, committed=False))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--json-out', default='logs/validate/spec_natural.json')
    parser.add_argument('--prompt-tokens', type=int, default=2048)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--draft-sweep', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--modes', nargs='+', choices=('sequential', 'packed', 'packed_guarded'),
                        default=['sequential', 'packed'])
    parser.add_argument('--cases', nargs='+', choices=tuple(natural_cases()), default=list(natural_cases()))
    parser.add_argument('--near-tie-margin', type=float, default=0.05)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, *args.draft_sweep) < 1:
        parser.error('lengths and draft limits must be positive')
    if (args.near_tie_margin < 0 or len(set(args.draft_sweep)) != len(args.draft_sweep)
            or len(set(args.modes)) != len(args.modes) or len(set(args.cases)) != len(args.cases)):
        parser.error('nonnegative margin and unique modes, cases and draft limits are required')
    engine = LLMEngine(args.model, max_num_seqs=2,
                       max_model_len=args.prompt_tokens+args.output_tokens+16,
                       max_num_batched_tokens=args.prompt_tokens, enforce_eager=True,
                       enable_prefix_cache=True, gpu_memory_utilization=0.6,
                       speculative=SpeculativeConfig(enabled=True))
    runner = engine.model_runner
    original_sample, original_verify = runner.sample_tokens, runner.verify_speculative
    original_logits = runner.model.compute_logits
    checkpoints = engine.scheduler.checkpoints
    records, failures = [], []
    current = {}

    def capture_logits(hidden):
        logits = original_logits(hidden)
        # Reference replays do not project logits; only actual prediction rows
        # are recorded. A near tie is diagnostic, never a relaxed acceptance rule.
        context = get_context()
        plan = current.get('plan')
        if plan is not None:
            positions = range(plan.computed_length+1, plan.computed_length+1+len(logits)) if context.is_prefill else [
                int(context.context_lens[0].item())]
            source = 'packed_trial' if context.is_prefill else 'decode_trial'
        else:
            endpoint = int((context.cu_seqlens_k[-1] if context.is_prefill
                            else context.context_lens[0]).item())
            positions = [endpoint]
            source = 'ordinary'
        rows = prediction_diagnostics(logits, positions, source)
        for row in rows:
            # Prefix checkpoints can split prefill before its final prompt
            # token. Those projected rows have emit=0 and are not outputs.
            row['committed'] = plan is None and row['position'] >= current['prompt_length']
        current['predictions'].extend(rows)
        return logits

    def capture_sample():
        seq = runner._pending[2][0]
        output = original_sample()
        if seq.num_completion_tokens+1 == args.output_tokens:
            current['final'] = snapshot(runner, seq)
        return output

    def capture_verify(seq, plan):
        before = snapshot(runner, seq)
        inactive = [(pool, pool[:runner.config.max_num_seqs].clone())
                    for layer in runner.gdn_layers for pool in (layer.conv_states, layer.recurrent_states)]
        prediction_start = len(current['predictions'])
        fallbacks_before = runner.spec_metrics.get('packed_fallbacks', 0)
        current['plan'] = plan
        try:
            result = original_verify(seq, plan)
        finally:
            current['plan'] = None
        actual = snapshot(runner, seq)
        used_reference = (current['mode'] == 'sequential'
                          or runner.spec_metrics.get('packed_fallbacks', 0) > fallbacks_before)
        source = 'decode_trial' if used_reference else 'packed_trial'
        for row in current['predictions'][prediction_start:]:
            row['committed'] = (row['source'] == source and row['position'] <= actual['computed'])
        if current['mode'] == 'packed':
            reference = same_block_reference(runner, seq, plan, result, before)
            errors = equal_state(reference, actual)
        else:
            errors = []
        slot = runner.input_batch.slots_for([seq])[0]
        for pool, saved in inactive:
            others = [i for i in range(runner.config.max_num_seqs) if i != slot]
            if not torch.equal(pool[others], saved[others]):
                errors.append('inactive_slot')
        if actual['computed'] != result.committed_computed_length:
            errors.append('gpu_endpoint')
        expected = (*seq.token_ids, *result.token_ids)
        if actual['tokens'].tolist() != list(expected):
            errors.append('gpu_history')
        if not torch.isfinite(actual['kv']).all():
            errors.append('nonfinite_kv')
        if not all(torch.isfinite(t).all() for key in ('conv', 'recurrent') for t in actual[key]):
            errors.append('nonfinite_state')
        current['checks'].append(dict(computed=actual['computed'], accepted=result.accepted_draft_tokens,
                                      output=result.output_length, mismatches=errors))
        if errors:
            raise AssertionError(f'committed state mismatch: {errors}')
        if result.finished:
            current['final'] = actual
        return result

    def run(prompt, mode, draft_limit):
        current.clear()
        current.update(mode=mode, prompt_length=len(prompt), predictions=[], checks=[], plan=None)
        config = None if mode == 'baseline' else SpeculativeConfig(
            enabled=True, verification_mode=mode, max_draft_tokens=draft_limit)
        engine.config.speculative = engine.scheduler.speculative = config
        before = dict(runner.spec_metrics)
        fallbacks = dict(engine.scheduler.spec_fallbacks)
        hits_before = checkpoints.hits
        tokens = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                                        ignore_eos=True), use_tqdm=False)[0]['token_ids']
        if len(tokens) != args.output_tokens or 'final' not in current:
            raise AssertionError('output length or final state missing')
        final = current['final']
        if final['computed'] != len(prompt)+len(tokens)-1:
            raise AssertionError('final computed endpoint differs from emitted history')
        if final['tokens'].tolist() != prompt+tokens:
            raise AssertionError('final GPU history differs from emitted history')
        committed_rows = [r for r in current['predictions'] if r['committed']]
        if ([r['position'] for r in committed_rows] != list(range(len(prompt), len(prompt)+len(tokens)))
                or [r['token'] for r in committed_rows] != tokens):
            raise AssertionError('prediction diagnostics do not align with actually emitted tokens')
        return dict(tokens=tokens, predictions=current['predictions'], state_checks=current['checks'],
                    prefix_cache_hits=checkpoints.hits-hits_before,
                    metrics={k: v-before.get(k, 0) for k, v in runner.spec_metrics.items()},
                    fallbacks={k: v-fallbacks.get(k, 0) for k, v in engine.scheduler.spec_fallbacks.items()}), final

    runner.sample_tokens, runner.verify_speculative = capture_sample, capture_verify
    runner.model.compute_logits = capture_logits
    completed = False
    try:
        for prefix in (False, True):
            engine.scheduler.enable_prefix_cache = prefix
            engine.scheduler.checkpoints = checkpoints if prefix else None
            for name, text in natural_cases().items():
                if name not in args.cases:
                    continue
                prompt = prepare_prompt(engine.tokenizer, text, args.prompt_tokens, 'truncate')
                baseline, base_state = run(prompt, 'baseline', 1)
                baseline_rows = {r['position']: r for r in baseline['predictions'] if r['committed']}
                for draft_limit in args.draft_sweep:
                    for mode in args.modes:
                        actual, final = run(prompt, mode, draft_limit)
                        mismatch = next((i for i, (a, b) in enumerate(zip(baseline['tokens'], actual['tokens']))
                                         if a != b), None)
                        near_ties = [r for r in actual['predictions'] if r['margin'] <= args.near_tie_margin]
                        first_row = next((r for r in actual['predictions']
                                          if r['committed'] and mismatch is not None
                                          and r['position'] == len(prompt)+mismatch), None)
                        drift = state_drift(base_state, final)
                        if mode in ('sequential', 'packed_guarded') and equal_state(base_state, final):
                            raise AssertionError('conservative mode differs from baseline committed state')
                        record = dict(case=name, prefix_cache=prefix, prompt_tokens=len(prompt), prompt_ids=prompt,
                                      output_tokens=len(actual['tokens']), mode=mode, max_draft_tokens=draft_limit,
                                      token_match=mismatch is None, first_token_mismatch=mismatch,
                                      first_mismatch_baseline=baseline_rows.get(len(prompt)+mismatch)
                                          if mismatch is not None else None,
                                      first_mismatch_trial=first_row, near_tie_predictions=near_ties,
                                      final_state_drift=drift, **actual)
                        records.append(record)
                        print(json.dumps({k: v for k, v in record.items()
                                          if k not in ('tokens', 'prompt_ids', 'predictions', 'state_checks', 'near_tie_predictions')}),
                              flush=True)
                        if mismatch is not None:
                            failures.append(dict(case=name, prefix_cache=prefix, mode=mode,
                                                 max_draft_tokens=draft_limit, position=mismatch))
        completed = True
    finally:
        runner.sample_tokens, runner.verify_speculative = original_sample, original_verify
        runner.model.compute_logits = original_logits
        reset_context()
        engine.exit()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(metadata=dict(model=args.model, torch=torch.__version__,
                                                     cuda=torch.version.cuda, precision=str(engine.config.hf_config.dtype),
                                                     gpu=torch.cuda.get_device_name(), prompt_limit=args.prompt_tokens,
                                                     output_tokens=args.output_tokens, near_tie_margin=args.near_tie_margin,
                                                     modes=args.modes, draft_limits=args.draft_sweep,
                                                     prediction_trace_checked=True,
                                                     prompt_normalization='truncate', scope='correctness diagnostics'),
                                        completed=completed, passed=completed and not failures,
                                        failures=failures, cases=records), indent=2)+'\n')
    if failures:
        raise AssertionError(f'{len(failures)} cases differ from ordinary greedy decoding; see {args.json_out}')


if __name__ == '__main__':
    main()
