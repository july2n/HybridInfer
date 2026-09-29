"""Compare cold and cached hybrid execution, including full logits/GDN tensors.

PYTHONPATH=src:.runtime-deps python benchmarks/validate_prefix_cache.py \
    --model models/Qwen3.5-0.8B --json-out logs/validate/prefix_cache.json
Use --graphs to validate CUDA Graph decode and piecewise prefill as well.
Captures synchronize CUDA and are for correctness, not timing measurements.
"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.engine.sequence import Sequence
from hybridinfer.sampling_params import SamplingParams
from engine_bench_utils import STATE_DRIFT_RMSE_LIMITS


class CaptureSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.records = {}

    def set_batch_context(self, seqs, is_prefill):
        self.rows = [(s.seq_id, s.num_completion_tokens,
                      s.num_cached_tokens + s.num_scheduled_tokens >= s.num_tokens)
                     for s in seqs]

    def observe_gdn_state(self, layers, slots):
        self.kv = []
        for slot in slots.tolist():
            length = int(self.runner.request_state.computed.tensor[slot])
            count = (length + self.runner.block_size - 1) // self.runner.block_size
            blocks = self.runner.request_state.block_tables.tensor[slot, :count].long()
            self.kv.append(self.runner.kv_cache.index_select(2, blocks)
                           .flatten(2, 3)[:, :, :length].detach().cpu())
        self.states = [
            (layer.conv_states.index_select(0, slots).detach().cpu(),
             layer.recurrent_states.index_select(0, slots).detach().cpu())
            for layer in layers
        ]

    def forward(self, logits, temperatures):
        for row, (seq_id, position, emit) in enumerate(self.rows):
            if emit:
                self.records.setdefault(seq_id, {})[position] = (
                    logits[row].detach().float().cpu(),
                    [(conv[row].clone(), recurrent[row].clone())
                     for conv, recurrent in self.states], self.kv[row],
                )
        return logits.argmax(dim=-1)


def compare(reference, candidate):
    result = dict(logits_max_abs=0., logits_rmse=0.,
                  conv_max_abs=0., recurrent_max_abs=0., kv_max_abs=0.,
                  conv_relative_rmse=0., recurrent_relative_rmse=0., kv_relative_rmse=0.)
    assert reference.keys() == candidate.keys(), 'missing output positions'
    for position in reference:
        logits_a, states_a, kv_a = reference[position]
        logits_b, states_b, kv_b = candidate[position]
        assert torch.isfinite(logits_b).all()
        delta = logits_a - logits_b
        result['logits_max_abs'] = max(result['logits_max_abs'], delta.abs().max().item())
        result['logits_rmse'] = max(result['logits_rmse'], delta.square().mean().sqrt().item())
        # Different batch shapes can perturb BF16 GEMMs; bound numerical drift.
        torch.testing.assert_close(logits_a, logits_b, rtol=.03, atol=.25)
        for (conv_a, recurrent_a), (conv_b, recurrent_b) in zip(states_a, states_b):
            for name, a, b in [('conv', conv_a, conv_b),
                               ('recurrent', recurrent_a, recurrent_b)]:
                assert torch.isfinite(b).all()
                result[name + '_max_abs'] = max(result[name + '_max_abs'],
                                               (a.float() - b.float()).abs().max().item())
                relative = ((a.float() - b.float()).square().mean().sqrt()
                            / a.float().square().mean().sqrt().clamp_min(1e-8)).item()
                result[name + '_relative_rmse'] = max(result[name + '_relative_rmse'], relative)
                assert relative <= STATE_DRIFT_RMSE_LIMITS[name], (name, position, relative)
        assert kv_a.shape == kv_b.shape and torch.isfinite(kv_b).all()
        kv_delta = kv_a.float() - kv_b.float()
        relative = (kv_delta.square().mean().sqrt()
                    / kv_a.float().square().mean().sqrt().clamp_min(1e-8)).item()
        result['kv_max_abs'] = max(result['kv_max_abs'], kv_delta.abs().max().item())
        result['kv_relative_rmse'] = max(result['kv_relative_rmse'], relative)
        assert relative <= STATE_DRIFT_RMSE_LIMITS['kv'], ('kv', position, relative)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--graphs', action='store_true')
    parser.add_argument('--compile-segments', action='store_true')
    parser.add_argument('--json-out', default='logs/validate/prefix_cache.json')
    args = parser.parse_args()
    if args.compile_segments and not args.graphs:
        parser.error('--compile-segments requires --graphs')
    engine = LLMEngine(args.model, enable_prefix_cache=True,
                       enable_piecewise_compile=args.compile_segments,
                       prefix_cache_num_snapshots=2, max_num_seqs=3,
                       max_model_len=768, max_num_batched_tokens=256,
                       enforce_eager=not args.graphs, gpu_memory_utilization=.75)
    sampler = CaptureSampler()
    engine.model_runner.sampler = sampler
    sampler.runner = engine.model_runner
    original_copy = engine.model_runner.kv_cache_manager.copy_prefix_state
    copies = {'restore': 0, 'save': 0}

    def checked_copy(seq, slot, restore):
        original_copy(seq, slot, restore)
        snapshot_id = seq.restore_snapshot_id if restore else seq.save_snapshot_id
        if snapshot_id is None:
            return
        for layer, (conv, recurrent) in zip(engine.model_runner.gdn_layers,
                                             engine.model_runner.prefix_snapshots):
            assert torch.equal(layer.conv_states[slot], conv[snapshot_id])
            assert torch.equal(layer.recurrent_states[slot], recurrent[snapshot_id])
        copies['restore' if restore else 'save'] += 1

    engine.model_runner.kv_cache_manager.copy_prefix_state = checked_copy
    pool = engine.scheduler.checkpoints
    generator = torch.Generator().manual_seed(91)
    prefix = torch.randint(100, 10000, (512,), generator=generator).tolist()
    prompts = [prefix + [120 + i] * 17 for i in range(2)]
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
    report = {'graphs': args.graphs, 'compile_segments': args.compile_segments, 'cases': {}}

    def run(prompts):
        seqs = [Sequence(p, params) for p in prompts]
        for seq in seqs:
            engine.scheduler.add(seq)
        while not engine.is_finished():
            engine.step()
        return [(s.completion_token_ids, sampler.records.pop(s.seq_id)) for s in seqs]

    def clear_checkpoints():
        assert not any(e.pins for e in pool.entries.values())
        pool.entries.clear()

    def check(name, baseline, cached, expected_hits):
        before_hits = pool.hits
        before_tokens = pool.hit_tokens
        outputs = run(cached)
        assert pool.hits - before_hits == expected_hits, name + ': unexpected cache hit count'
        assert pool.hit_tokens - before_tokens == expected_hits * 512
        diagnostics = []
        for (tokens_a, states_a), (tokens_b, states_b) in zip(baseline, outputs):
            assert tokens_a == tokens_b, name + ': generated token mismatch'
            diagnostics.append(compare(states_a, states_b))
        report['cases'][name] = dict(hits=expected_hits,
                                    hit_tokens=pool.hit_tokens - before_tokens,
                                    diagnostics=diagnostics)

    try:
        # Cold paths use the same chunk boundaries, but no state restoration.
        clear_checkpoints()
        baseline_a = run([prompts[0]])
        check('exact_repeat', baseline_a, [prompts[0]], 1)
        clear_checkpoints()
        baseline_b = run([prompts[1]])
        clear_checkpoints()
        run([prompts[0]])
        check('different_suffix', baseline_b, [prompts[1]], 1)
        # Preserve immutable snapshots across simultaneous forked continuations.
        immutable = [(a.clone(), b.clone()) for a, b in engine.model_runner.prefix_snapshots]
        check('concurrent_forks', baseline_a + baseline_b, prompts, 2)
        for (a, b), (old_a, old_b) in zip(engine.model_runner.prefix_snapshots, immutable):
            assert torch.equal(a, old_a) and torch.equal(b, old_b), 'snapshot was mutated'
        # Evict checkpoints while leaving old KV in the block cache.
        clear_checkpoints()
        check('snapshot_evicted', baseline_a, [prompts[0]], 0)
        # Evict every idle KV payload while leaving GDN checkpoints intact.
        bm = engine.scheduler.block_manager
        assert not bm.used_block_ids
        blocks = [bm._allocate_block() for _ in range(len(bm.free_block_ids))]
        for block_id in blocks:
            bm.blocks[block_id].ref_count = 0
            bm._deallocate_block(block_id)
        check('kv_evicted', baseline_a, [prompts[0]], 0)
        report['exact_state_copies'] = copies
        report['passed'] = True
    finally:
        engine.exit()
    path = Path(args.json_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
