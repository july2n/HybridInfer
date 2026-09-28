"""Offline single-request n-gram benchmark, without state snapshots or oracle drafts."""
import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from spec_workloads import natural_cases, prepare_prompt


CASES = {
    'repetition': 'Continue repeating alpha beta gamma delta, with no explanation:\n' + 'alpha beta gamma delta\n'*32,
    'code': 'Continue this Python file with functions following the same pattern:\n' + '\n'.join(
        f'def add_{i}(x):\n    return x + {i}\n' for i in range(24)),
    'dialogue': 'Explain how a CPU cache works, with an example of traversing a matrix. Discuss locality and cache misses.',
}


def summarize_samples(samples, baseline):
    if not samples or len(samples) != len(baseline):
        raise ValueError('each measurement requires a paired baseline')
    seconds = statistics.median(s['decode_seconds'] for s in samples)
    base_seconds = statistics.median(s['decode_seconds'] for s in baseline)
    drafts = sum(s['metrics']['draft_tokens'] for s in samples)
    accepts = sum(s['metrics']['accepted_tokens'] for s in samples)
    matches = [s['tokens'] == base['tokens'] for s, base in zip(samples, baseline)]
    first_mismatches = [next((i for i, (a, b) in enumerate(zip(base['tokens'], s['tokens'])) if a != b),
                            min(len(base['tokens']), len(s['tokens']))
                                if len(base['tokens']) != len(s['tokens']) else None)
                        for s, base in zip(samples, baseline)]
    paired = [base['decode_seconds']/s['decode_seconds'] for s, base in zip(samples, baseline)]
    comparable = all(matches)
    return dict(median_decode_tokens_per_second=statistics.median(s['decode_tokens_per_second'] for s in samples),
                median_generation_seconds=statistics.median(s['generation_seconds'] for s in samples),
                speedup_comparable=comparable, measured_decode_time_ratio=base_seconds/seconds,
                decode_speedup=base_seconds/seconds if comparable else None,
                paired_decode_speedups=paired,
                median_paired_decode_speedup=statistics.median(paired) if comparable else None,
                draft_acceptance_rate=accepts/drafts if drafts else None,
                baseline_token_matches=matches, first_token_mismatches=first_mismatches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--json-out', default='logs/bench/spec_decode.json')
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--prompt-tokens', type=int, default=509)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--draft-tokens', type=int, default=4)
    parser.add_argument('--enforce-eager', action='store_true')
    parser.add_argument('--suite', choices=('synthetic', 'natural'), default='synthetic')
    parser.add_argument('--draft-sweep', type=int, nargs='+', help='Candidate limits, e.g. 1 2 4 8')
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'sequential', 'packed', 'packed_guarded'),
                        default=['baseline', 'packed'])
    parser.add_argument('--require-token-match', action='store_true',
                        help='Exit nonzero after saving results if any outputs differ')
    args = parser.parse_args()
    if min(args.output_tokens, args.prompt_tokens, args.repeats, args.warmups, args.draft_tokens) < 1:
        parser.error('lengths, repeats and warmups must be positive')
    draft_limits = args.draft_sweep or [args.draft_tokens]
    if (min(draft_limits) < 1 or len(set(draft_limits)) != len(draft_limits)
            or 'baseline' not in args.modes or len(set(args.modes)) != len(args.modes)):
        parser.error('positive draft limits and unique modes including baseline are required')
    normalization = 'truncate' if args.suite == 'natural' else 'repeat_and_truncate'
    cases = natural_cases() if args.suite == 'natural' else CASES
    engine = LLMEngine(args.model, max_num_seqs=2,
                       max_model_len=args.prompt_tokens+args.output_tokens+8,
                       max_num_batched_tokens=max(512, args.prompt_tokens),
                       enforce_eager=args.enforce_eager, use_prefill_cudagraph=False,
                       gpu_memory_utilization=0.6, enable_prefix_cache=False,
                       speculative=SpeculativeConfig(enabled=True))
    records = []
    def run(prompt, mode, draft_limit):
        config = SpeculativeConfig(enabled=True, verification_mode=mode,
                                   max_draft_tokens=draft_limit) if mode != 'baseline' else None
        engine.config.speculative = engine.scheduler.speculative = config
        engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True))
        seq = engine.scheduler.waiting[-1]
        before = dict(engine.model_runner.spec_metrics)
        fallback_before = dict(engine.scheduler.spec_fallbacks)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        while not engine.is_finished() and not engine.scheduler.running:
            engine.step()
        torch.cuda.synchronize()
        prefill_end = perf_counter()
        generated_before = seq.num_completion_tokens
        while not engine.is_finished():
            engine.step()
        torch.cuda.synchronize()
        end = perf_counter()
        metrics = {key: value-before.get(key, 0) for key, value in engine.model_runner.spec_metrics.items()}
        fallbacks = {key: value-fallback_before.get(key, 0) for key, value in engine.scheduler.spec_fallbacks.items()}
        count = seq.num_completion_tokens-generated_before
        return dict(prefill_seconds=prefill_end-start, decode_seconds=end-prefill_end,
                    generation_seconds=end-start, decode_output_tokens=count,
                    decode_tokens_per_second=count/(end-prefill_end), metrics=metrics,
                    scheduler_fallbacks=fallbacks,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), tokens=seq.completion_token_ids)
    completed = False
    try:
        for name, text in cases.items():
            prompt = prepare_prompt(engine.tokenizer, text, args.prompt_tokens, normalization)
            for draft_limit in draft_limits:
                modes = args.modes
                for _ in range(args.warmups):
                    for mode in modes:
                        run(prompt, mode, draft_limit)
                samples_by_mode = {mode: [] for mode in modes}
                orders = []
                # Rotate the first mode so clock/thermal drift does not always
                # favor the last configuration. Pair results within each round.
                for repeat in range(args.repeats):
                    shift = repeat % len(modes)
                    order = modes[shift:] + modes[:shift]
                    orders.append(order)
                    paired = {mode: run(prompt, mode, draft_limit) for mode in order}
                    for mode in modes:
                        samples_by_mode[mode].append(paired[mode])
                baseline = samples_by_mode['baseline']
                for mode in modes:
                    samples = samples_by_mode[mode]
                    record = dict(case=name, mode=mode, max_draft_tokens=draft_limit,
                                  prompt_ids=prompt, samples=samples, measurement_orders=orders,
                                  **summarize_samples(samples, baseline))
                    records.append(record)
                    print(json.dumps({key: value for key, value in record.items()
                                      if key not in ('samples', 'prompt_ids')}), flush=True)
                    if any(len(s['tokens']) != args.output_tokens for s in samples):
                        raise AssertionError('wrong output length')
        completed = True
    finally:
        metadata = dict(model=str(Path(args.model).resolve()), torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(), precision=str(engine.config.hf_config.dtype),
                        batch_size=1, prefix_cache=False, temperature=0, ignore_eos=True,
                        decode_graphs=not args.enforce_eager, native_verification='eager',
                        warmups_per_case_mode=args.warmups, repeats=args.repeats,
                        prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                        prompt_normalization=normalization, suite=args.suite,
                        measurement_order='rotating_interleaved', modes=args.modes,
                        enabled_default_verification=SpeculativeConfig().verification_mode,
                        max_draft_tokens=draft_limits, token_match_required=args.require_token_match,
                        scope='offline decode and whole generation')
        engine.exit()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(metadata=metadata, completed=completed,
                                       all_token_matches=completed and bool(records)
                                           and all(all(c['baseline_token_matches']) for c in records),
                                       cases=records), indent=2)+'\n')
    if args.require_token_match and any(not all(c['baseline_token_matches']) for c in records):
        raise AssertionError(f'outputs differ from baseline; see {args.json_out}')


if __name__ == '__main__':
    main()
