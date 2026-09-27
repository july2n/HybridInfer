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


CASES = {
    'repetition': 'Continue repeating alpha beta gamma delta, with no explanation:\n' + 'alpha beta gamma delta\n'*32,
    'code': 'Continue this Python file with functions following the same pattern:\n' + '\n'.join(
        f'def add_{i}(x):\n    return x + {i}\n' for i in range(24)),
    'dialogue': 'Explain how a CPU cache works, with an example of traversing a matrix. Discuss locality and cache misses.',
}


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
    args = parser.parse_args()
    if min(args.output_tokens, args.prompt_tokens, args.repeats, args.warmups, args.draft_tokens) < 1:
        parser.error('lengths, repeats and warmups must be positive')
    engine = LLMEngine(args.model, max_num_seqs=2,
                       max_model_len=args.prompt_tokens+args.output_tokens+8,
                       max_num_batched_tokens=max(512, args.prompt_tokens),
                       enforce_eager=args.enforce_eager, use_prefill_cudagraph=False,
                       gpu_memory_utilization=0.6, enable_prefix_cache=False,
                       speculative=SpeculativeConfig(enabled=True))
    records = []
    def run(prompt, mode):
        config = SpeculativeConfig(enabled=True, verification_mode=mode,
                                   max_draft_tokens=args.draft_tokens) if mode != 'baseline' else None
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
    try:
        for name, text in CASES.items():
            raw = engine.tokenizer.encode(text)
            prompt = (raw*(args.prompt_tokens//len(raw)+1))[:args.prompt_tokens]
            baseline_tokens = None
            baseline_seconds = None
            for mode in ('baseline', 'sequential', 'packed'):
                for _ in range(args.warmups):
                    run(prompt, mode)
                samples = [run(prompt, mode) for _ in range(args.repeats)]
                if mode == 'baseline':
                    baseline_tokens = samples[0]['tokens']
                    baseline_seconds = statistics.median(s['decode_seconds'] for s in samples)
                median_seconds = statistics.median(s['decode_seconds'] for s in samples)
                drafts = sum(s['metrics']['draft_tokens'] for s in samples)
                accepts = sum(s['metrics']['accepted_tokens'] for s in samples)
                first_mismatches = [next((i for i, (a, b) in enumerate(zip(baseline_tokens, s['tokens'])) if a != b), None)
                                    for s in samples]
                record = dict(case=name, mode=mode, prompt_ids=prompt, samples=samples,
                              median_decode_tokens_per_second=statistics.median(s['decode_tokens_per_second'] for s in samples),
                              median_generation_seconds=statistics.median(s['generation_seconds'] for s in samples),
                              decode_speedup=baseline_seconds/median_seconds,
                              draft_acceptance_rate=accepts/drafts if drafts else None,
                              baseline_token_matches=[s['tokens'] == baseline_tokens for s in samples],
                              first_token_mismatches=first_mismatches)
                records.append(record)
                print(json.dumps({key: value for key, value in record.items() if key not in ('samples', 'prompt_ids')}), flush=True)
                if any(len(s['tokens']) != args.output_tokens for s in samples):
                    raise AssertionError('wrong output length')
    finally:
        metadata = dict(model=str(Path(args.model).resolve()), torch=torch.__version__, cuda=torch.version.cuda,
                        gpu=torch.cuda.get_device_name(), precision=str(engine.config.hf_config.dtype),
                        batch_size=1, prefix_cache=False, temperature=0, ignore_eos=True,
                        decode_graphs=not args.enforce_eager, native_verification='eager',
                        warmups_per_case_mode=args.warmups, repeats=args.repeats,
                        prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens,
                        prompt_normalization='repeat_and_truncate',
                        max_draft_tokens=args.draft_tokens, scope='offline decode and whole generation')
        engine.exit()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(metadata=metadata, cases=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
