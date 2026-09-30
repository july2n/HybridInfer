"""Compare warmed engine inference with NONE and FULL_AND_PIECEWISE.

Run each mode in a separate process; timing includes scheduling and sampling.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
import time

from bench_one_batch import find_repo, enqueue, full_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--mode', choices=['NONE', 'FULL_AND_PIECEWISE'], required=True)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16, 32])
    parser.add_argument('--input-len', type=int, default=64)
    parser.add_argument('--decode-steps', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--json-out', required=True)
    args = parser.parse_args()
    find_repo()
    import torch
    from hybridinfer.engine.llm_engine import LLMEngine
    from hybridinfer.engine.sequence import Sequence
    from hybridinfer.sampling_params import SamplingParams

    torch.manual_seed(42)
    started = time.perf_counter()
    engine = LLMEngine(args.model, cudagraph_mode=args.mode,
                       enable_piecewise_compile=False, enable_prefix_cache=False,
                       max_num_seqs=max(args.batches), max_num_batched_tokens=max(args.batches)*args.input_len,
                       max_model_len=512, gpu_memory_utilization=.65)
    torch.cuda.synchronize()
    result = {'mode': args.mode, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
              'args': vars(args), 'init_s': time.perf_counter()-started, 'compile': False,
              'queue_depth': engine.max_concurrent_batches, 'async_output': engine.model_runner.async_output,
              'decode_buckets': engine.model_runner.cuda_graphs.decode_graph_sizes,
              'piecewise_buckets': engine.model_runner.cuda_graphs.prefill_graph_sizes, 'cases': []}
    counts = Counter()
    dispatcher = engine.model_runner.cuda_graphs.dispatcher
    original = dispatcher.dispatch
    def dispatch(*a, **kw):
        mode = original(*a, **kw)
        counts[mode.value] += 1
        return mode
    dispatcher.dispatch = dispatch
    try:
        with torch.inference_mode():
            for batch in args.batches:
                workload = SimpleNamespace(batch_size=batch, input_len=args.input_len, decode_steps=args.decode_steps)
                samples = []
                for i in range(args.warmup+args.repeat):
                    counts.clear()
                    torch.manual_seed(42+i)
                    seqs = enqueue(engine, workload, Sequence, SamplingParams, 42+i)
                    timing = full_run(engine, seqs, workload, torch)
                    if i >= args.warmup:
                        samples.append(timing)
                case = {'batch_size': batch, 'median': {k: statistics.median(s[k] for s in samples) for k in samples[0]},
                        'samples': samples, 'last_path_counts': dict(counts)}
                result['cases'].append(case)
                print(json.dumps(case), flush=True)
    finally:
        engine.exit()
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
