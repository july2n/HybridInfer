"""Isolate padded-vs-packed conv and gathered-vs-indexed FP32 GDN decode.

PYTHONPATH=src:.runtime-deps python benchmarks/bench_gdn_state_pool.py \
    --json-out logs/bench/gdn_state_pool.json
Reports CUDA event time and synchronized host wall time. No model loading.
"""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
import torch.nn.functional as F
from hybridinfer.layers.gdn_kernels import packed_causal_conv, indexed_gdn_decode
from hybridinfer.layers.gated_delta_net import decode_gated_delta_rule


def measure(fn, rounds):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    gpu, wall = [], []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = perf_counter()
        start.record()
        fn()
        end.record()
        end.synchronize()
        wall.append((perf_counter() - before) * 1000)
        gpu.append(start.elapsed_time(end))
    return {'gpu_ms': median(gpu), 'wall_ms': median(wall)}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--json-out', default='logs/bench/gdn_state_pool.json')
    args = parser.parse_args()
    torch.manual_seed(27)
    report = {'device': torch.cuda.get_device_name(), 'rounds': args.rounds, 'conv': {}, 'decode': {}}
    for lengths in ([512], [128] * 4, [64, 128, 256, 512], [1] * 8):
        channels, width = 6144, 3
        x = torch.randn(sum(lengths), channels, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(channels, 1, width+1, device='cuda', dtype=torch.bfloat16)
        pool = torch.zeros(len(lengths), channels, width, device='cuda', dtype=torch.bfloat16)
        slots = torch.arange(len(lengths), device='cuda')
        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        cu = torch.tensor(bounds, device='cuda', dtype=torch.int32)

        def padded():
            padded = x.new_zeros(len(lengths), channels, max(lengths) + width)
            padded[:, :, :width] = pool.index_select(0, slots)
            for req, length in enumerate(lengths):
                padded[req, :, width:width+length] = x[bounds[req]:bounds[req+1]].T
            conv = F.silu(F.conv1d(padded, w, groups=channels))
            output = torch.cat([conv[i, :, :length].T for i, length in enumerate(lengths)])
            histories = torch.stack([padded[i, :, :width+length][:, -width:] for i, length in enumerate(lengths)])
            pool.index_copy_(0, slots, histories)
            return output

        old = measure(padded, args.rounds)
        new = measure(lambda: packed_causal_conv(x, w, pool, slots, cu, max(lengths)), args.rounds)
        report['conv'][str(lengths)] = {'padded': old, 'packed': new, 'gpu_speedup': old['gpu_ms']/new['gpu_ms']}
    for batch in (1, 3, 8, 16):
        q = torch.randn(batch, 1, 16, 128, device='cuda', dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        a = torch.randn(batch, 1, 16, device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        log = torch.randn(16, device='cuda')
        bias = torch.randn(16, device='cuda', dtype=torch.bfloat16)
        pool = torch.zeros(batch+2, 16, 128, 128, device='cuda')
        slots = torch.arange(batch, device='cuda').flip(0) + 1

        def gathered():
            output, state = decode_gated_delta_rule(q, k, v, a, b, log, bias, pool.index_select(0, slots))
            pool.index_copy_(0, slots, state)
            return output

        old = measure(gathered, args.rounds)
        new = measure(lambda: indexed_gdn_decode(q, k, v, a, b, log, bias, pool, slots), args.rounds)
        report['decode'][str(batch)] = {'gathered_flashinfer': old, 'indexed': new, 'gpu_speedup': old['gpu_ms']/new['gpu_ms']}
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
