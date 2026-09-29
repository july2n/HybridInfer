"""Fullgraph compilation experiments on real Qwen3.5 weights, without changing runtime dispatch."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten

from hybridinfer.config import Config
from hybridinfer.engine.model_runner import find_free_port
from hybridinfer.models.qwen3_5 import Qwen3_5ForCausalLM
from hybridinfer.utils.loader import load_model


def timing(fn, args, repeats):
    for _ in range(5):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn(*args)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def graph_timing(fn, args, repeats):
    # Compilation and allocations are warmed before stream capture.
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn(*args)  # Retain captured output storage until replay finishes.
    elapsed = timing(lambda: graph.replay(), (), repeats)
    return elapsed


def compare(a, b):
    aa, sa = tree_flatten(a)
    bb, sb = tree_flatten(b)
    assert sa == sb
    result = []
    for x, y in zip(aa, bb):
        assert x.shape == y.shape and x.dtype == y.dtype
        assert torch.isfinite(y).all()
        delta = x.float() - y.float()
        result.append(dict(shape=list(x.shape), exact=torch.equal(x, y),
                           max_abs=delta.abs().max().item(),
                           relative_rmse=(delta.square().mean().sqrt() /
                                          x.float().square().mean().sqrt().clamp_min(1e-8)).item()))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--tokens', nargs='+', type=int, default=[1, 128, 512])
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--json-out', default='logs/compile/segments.json')
    args = parser.parse_args()
    if args.repeats <= 0 or any(n <= 0 for n in args.tokens):
        parser.error("--repeats and --tokens must be positive")
    config = Config(args.model)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{find_free_port()}', rank=0, world_size=1)
    report = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  model=args.model, fullgraph=True, dynamic=False,
                  inductor_cudagraphs=False, input_kind='random hidden states, real checkpoint weights', cases=[])
    default_dtype = torch.get_default_dtype()
    try:
        with torch.device('cuda'):
            torch.set_default_dtype(config.hf_config.dtype)
            model = Qwen3_5ForCausalLM(config.hf_config).eval()
            load_model(model, args.model)
        layers = {}
        for index, layer in enumerate(model.model.layers):
            layers.setdefault(layer.block_type, (index, layer))
        for n in args.tokens:
            hidden = torch.randn(n, config.hf_config.hidden_size, device='cuda', dtype=config.hf_config.dtype)
            residual = torch.randn_like(hidden)
            candidates = []
            for kind, (index, layer) in layers.items():
                candidates.extend([(f'{kind}.pre.residual', index, layer.forward_piecewise_pre, (hidden, residual)),
                                   (f'{kind}.pre.no_residual', index, layer.forward_piecewise_pre, (hidden, None)),
                                   (f'{kind}.post', index, layer.forward_output, (hidden, residual))])
            index, attention_layer = layers['full_attention']
            attn = attention_layer.self_attn
            q = torch.randn(n, attn.num_heads, attn.head_dim, device='cuda', dtype=hidden.dtype)
            k = torch.randn(n, attn.num_kv_heads, attn.head_dim, device='cuda', dtype=hidden.dtype)
            candidates.append(('mrope', index, attn.rotary_emb.forward,
                               (torch.arange(n, device='cuda'), q, k)))
            for name, index, fn, inputs in candidates:
                torch._dynamo.reset()
                case = dict(component=name, layer=index, tokens=n)
                try:
                    graphs = []
                    def backend(gm, example_inputs, **kwargs):
                        graphs.append(len(list(gm.graph.nodes)))
                        return torch._inductor.compile(gm, example_inputs, options=kwargs.get("options"))
                    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False,
                                             options={'triton.cudagraphs': False})
                    reference = fn(*inputs)
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    actual = compiled(*inputs)
                    torch.cuda.synchronize()
                    case['first_call_seconds'] = time.perf_counter() - started
                    case['outputs'] = compare(reference, actual)
                    case['eager_ms'] = timing(fn, inputs, args.repeats)
                    case['compiled_ms'] = timing(compiled, inputs, args.repeats)
                    case['speedup'] = case['eager_ms'] / case['compiled_ms']
                    case['eager_graph_ms'] = graph_timing(fn, inputs, args.repeats)
                    case['compiled_graph_ms'] = graph_timing(compiled, inputs, args.repeats)
                    case['graph_speedup'] = case['eager_graph_ms'] / case['compiled_graph_ms']
                    case['captured_graphs'] = len(graphs)
                    case['graph_nodes'] = graphs
                    case['status'] = 'ok'
                except Exception as exc:
                    case.update(status='error', error_type=type(exc).__name__, error=str(exc)[:4000])
                report['cases'].append(case)
                path = Path(args.json_out)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps({k:v for k,v in case.items() if k not in ('outputs', 'error')}, ensure_ascii=False), flush=True)
                if case['status'] == 'error':
                    print(case['error'], flush=True)
    finally:
        torch.set_default_dtype(default_dtype)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
