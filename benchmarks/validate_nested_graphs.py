"""Same-input eager versus nested-wrapper logits and cache-state diagnostics."""
import argparse
import json
from pathlib import Path

import torch

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.engine.sequence import Sequence
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.utils.context import get_context


def drift(reference, candidate):
    if not torch.isfinite(candidate).all():
        raise AssertionError('nonfinite compiled output/state')
    a, b = reference.float(), candidate.float()
    return dict(exact=torch.equal(reference, candidate), max_abs=(a-b).abs().max().item(),
                relative_rmse=((a-b).square().mean().sqrt() /
                               a.square().mean().sqrt().clamp_min(1e-8)).item())


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-segments', action='store_true')
    parser.add_argument('--mode', default='FULL_AND_PIECEWISE',
                        choices=['NONE', 'FULL', 'PIECEWISE', 'FULL_AND_PIECEWISE'])
    parser.add_argument('--json-out', default='logs/compile/nested_same_input.json')
    args = parser.parse_args()
    engine = LLMEngine('models/Qwen3.5-0.8B', max_num_seqs=4, max_model_len=512,
                       max_num_batched_tokens=128, gpu_memory_utilization=.5,
                       enable_piecewise_compile=args.compile_segments, cudagraph_mode=args.mode)
    runner = engine.model_runner
    original = runner.run_model
    report = dict(torch=torch.__version__, mode=args.mode, compile_segments=args.compile_segments,
                  scope='same-input local numerical diagnostics, not quality acceptance', cases=[])
    seen = set()

    def probe(ids, positions, prefill):
        ctx = get_context()
        counts = [b-a for a,b in ctx.prefill_slices] if prefill else [1]*ids.numel()
        label = 'mixed' if prefill and len(set(counts)) > 1 else 'prefill' if prefill else 'decode'
        key = (label, ctx.block_tables is not None)
        if key in seen:
            return original(ids, positions, prefill)
        seen.add(key)
        pages = runner.request_state.block_tables.tensor.index_select(
            0, runner.batch_slots_gpu[:ctx.batch_descriptor.num_reqs]).flatten()
        pages = pages[pages >= 0].unique().long()
        def snapshot():
            return (runner.kv_cache.index_select(2, pages).clone(),
                    [(layer.conv_states.clone(), layer.recurrent_states.clone()) for layer in runner.gdn_layers])
        def restore(state):
            runner.kv_cache.index_copy_(2, pages, state[0])
            for layer, (conv, rec) in zip(runner.gdn_layers, state[1]):
                layer.conv_states.copy_(conv)
                layer.recurrent_states.copy_(rec)
        def written():
            return runner.kv_cache.flatten(2, 3).index_select(2, ctx.slot_mapping.long()).clone()
        before = snapshot()
        reference = runner.compute_logits(runner.model(ids, positions), prefill)
        after = snapshot()
        expected_kv = written()
        restore(before)
        try:
            result = original(ids, positions, prefill)
            actual = snapshot()
            a, b = reference.float().softmax(-1), result.float().softmax(-1)
            case = dict(batch=label, query_lengths=counts, paged=ctx.block_tables is not None,
                        runtime_mode=runner.cuda_graphs.dispatcher.dispatch(
                            ctx.batch_descriptor, prefill, ids.numel()).value,
                        logits=drift(reference, result), kv=drift(expected_kv, written()),
                        states=[dict(conv=drift(x[0], y[0]), recurrent=drift(x[1], y[1]))
                                for x,y in zip(after[1], actual[1])],
                        mean_tv=((a-b).abs().sum(-1)*.5).mean().item(),
                        argmax_flips=(a.argmax(-1)!=b.argmax(-1)).sum().item())
            report['cases'].append(case)
            print(json.dumps({k:v for k,v in case.items() if k != 'states'}), flush=True)
        finally:
            restore(after)
        return reference  # Follow eager history for subsequent comparisons.

    runner.run_model = probe
    def add(n):
        seq = Sequence([100+i%100 for i in range(n)],
                       SamplingParams(temperature=0, max_tokens=3, ignore_eos=True))
        engine.scheduler.add(seq)
    def step():
        batch, prefill = engine.scheduler.schedule()
        if not batch:
            raise AssertionError('no runnable batch')
        runner.execute_model(batch, prefill)
        tokens = runner.sample_tokens().get_output()
        engine.scheduler.postprocess(batch, tokens, prefill)
        for seq in batch:
            if seq.is_finished:
                runner.remove_request(seq.seq_id)
    def drain():
        while not engine.scheduler.is_finished():
            step()
    try:
        add(200)
        drain()
        add(8)
        step()
        add(16)
        drain()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+'\n')
    finally:
        runner.run_model = original
        engine.exit()


if __name__ == '__main__':
    main()
