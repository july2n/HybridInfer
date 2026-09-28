"""Teacher-forced, state-controlled single-token vs packed target experiments.

Benchmark-only interventions; no proposer, acceptance, or production defaults.
Run with PYTHONPATH=src:.runtime-deps in the vllm_env Python environment.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import re
from pathlib import Path

import torch
import triton
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.layers import gated_delta_net as gdn
from hybridinfer.layers import gdn_kernels as kernels
from hybridinfer.layers.attention import Attention, store_kvcache, flash_attn_with_kvcache
from hybridinfer.layers.linear import LinearBase
from hybridinfer.layers.layernorm import GemmaRMSNorm, RMSNormGated
from hybridinfer.layers.activation import SiluAndMul
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.spec_decode.execution import packed_forward
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.utils.context import BatchDescriptor, get_context, reset_context, set_context


def difference(reference, candidate):
    a, b = reference.float(), candidate.float()
    d = b-a
    return dict(equal=torch.equal(reference, candidate), max_abs=d.abs().max().item(),
                rms=d.square().mean().sqrt().item(),
                relative_l2=(d.norm()/a.norm().clamp_min(1e-30)).item(),
                finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))


@contextmanager
def intervention(model, name, norm_audit=None):
    """All overrides are restored even when an experiment fails."""
    flags = set(name.split('+'))
    old_conv, old_rec = gdn.packed_causal_conv, gdn.packed_gdn_recurrent
    originals = []
    try:
        if 'conv' in flags:
            def conv(*args, **kwargs):
                kwargs['round_before_silu'] = True
                return old_conv(*args, **kwargs)
            gdn.packed_causal_conv = conv
        if flags & {'recurrent', 'bv', 'fma'}:
            bv = 8 if flags & {'recurrent', 'bv'} else 32
            fusion = not bool(flags & {'recurrent', 'fma'})
            def recurrent(q, k, v, a, b, log, bias, pool, slots, cu):
                total, hq, dk = q.shape
                hv, dv = v.shape[1:]
                out = torch.empty_like(v)
                states = torch.empty((total, hv, dv, dk), device=pool.device, dtype=pool.dtype)
                kernels._packed_recurrent[(slots.numel(), hv, triton.cdiv(dv, bv))](
                    q, k, v, a, b, log, bias, pool, slots, cu, out, states,
                    hq, hv, dk, dv, triton.next_power_of_2(dk), bv,
                    enable_fp_fusion=fusion, num_stages=3, num_warps=4)
                return out, states
            gdn.packed_gdn_recurrent = recurrent
        if 'gemm' in flags:
            for module in model.modules():
                if isinstance(module, (torch.nn.Linear, LinearBase)):
                    original = module.forward
                    originals.append((module, original))
                    def rowwise(x, original=original):
                        shape = x.shape
                        rows = x.reshape(-1, shape[-1])
                        y = torch.cat([original(row[None]) for row in rows], dim=0)
                        return y.reshape(*shape[:-1], y.shape[-1])
                    module.forward = rowwise
        if 'attention' in flags:
            for module in model.modules():
                if isinstance(module, Attention):
                    original = module.forward
                    originals.append((module, original))
                    def decode_attention(q, k, v, module=module, original=original):
                        ctx = get_context()
                        if not ctx.is_prefill:
                            return original(q, k, v)
                        if ctx.cu_seqlens_q.numel() != 2:
                            raise ValueError('Diagnostic attention supports one request only')
                        store_kvcache(k, v, module.k_cache, module.v_cache, ctx.slot_mapping)
                        start = int(ctx.cu_seqlens_k[-1])-q.shape[0]
                        rows = [flash_attn_with_kvcache(q[i:i+1, None], module.k_cache,
                            module.v_cache, cache_seqlens=torch.tensor([start+i+1],
                            device=q.device, dtype=torch.int32), block_table=ctx.block_tables,
                            softmax_scale=module.scale, causal=True).reshape(1, *q.shape[1:])
                            for i in range(q.shape[0])]
                        return torch.cat(rows)
                    module.forward = decode_attention
        if 'norm_mean' in flags:
            for module in model.modules():
                if isinstance(module, GemmaRMSNorm):
                    original = module.forward
                    originals.append((module, original))
                    def mean_matched(x, residual=None, module=module):
                        count = min(get_context().batch_descriptor.num_tokens, x.shape[0])
                        if x.shape[0] % count:
                            raise ValueError('Unexpected mean token layout')
                        combined = x if residual is None else x+residual
                        y = combined.float()
                        square = y.pow(2)
                        group = x.shape[0]//count
                        variance = torch.cat([square[i*group:(i+1)*group].mean(-1, keepdim=True)
                                              for i in range(count)])
                        y = y*torch.rsqrt(variance+module.eps)
                        y = (y*(1.0+module.weight.float())).to(x.dtype)
                        return y if residual is None else (y, combined)
                    module.forward = mean_matched
        norm_patterns = [re.compile(flag[5:]) for flag in flags if flag.startswith('norm:')]
        if flags & {'pointwise', 'gemma_norm', 'gated_norm', 'activation'} or norm_patterns or norm_audit is not None:
            for module_name, module in model.named_modules():
                selected = ('pointwise' in flags and isinstance(module, (GemmaRMSNorm, RMSNormGated, SiluAndMul))
                    or 'gemma_norm' in flags and isinstance(module, GemmaRMSNorm)
                    or 'gated_norm' in flags and isinstance(module, RMSNormGated)
                    or 'activation' in flags and isinstance(module, SiluAndMul)
                    or isinstance(module, GemmaRMSNorm) and any(p.fullmatch(module_name) for p in norm_patterns))
                audited = norm_audit is not None and isinstance(module, GemmaRMSNorm)
                if selected or audited:
                    original = module.forward
                    originals.append((module, original))
                    def tokenwise(x, *args, original=original, selected=selected,
                                  audited=audited, module=module, module_name=module_name):
                        count = get_context().batch_descriptor.num_tokens
                        if count == 1:
                            return original(x, *args)
                        if x.shape[0] % count:
                            raise ValueError('Unexpected pointwise token layout')
                        group = x.shape[0]//count
                        values = [original(x[i*group:(i+1)*group],
                            *(a[i*group:(i+1)*group] if a is not None else None for a in args))
                            for i in range(count)]
                        split = (tuple(torch.cat([v[j] for v in values]) for j in range(len(values[0])))
                                 if isinstance(values[0], tuple) else torch.cat(values))
                        if audited:
                            packed = original(x, *args)
                            p = packed[0] if isinstance(packed, tuple) else packed
                            s = split[0] if isinstance(split, tuple) else split
                            stats = difference(p, s)
                            entry = dict(module=module_name, input_shape=list(x.shape), output=stats,
                                differing_query_rows=(p != s).reshape(count, -1).any(-1).nonzero().flatten().tolist())
                            if not stats['equal']:
                                entry['operands'] = dict(x=x.detach().cpu(),
                                    residual=args[0].detach().cpu() if args and args[0] is not None else None,
                                    weight=module.weight.detach().cpu(), eps=module.eps,
                                    packed=p.detach().cpu(), rowwise=s.detach().cpu())
                            norm_audit.append(entry)
                            return split if selected else packed
                        return split
                    module.forward = tokenwise
        yield
    finally:
        gdn.packed_causal_conv, gdn.packed_gdn_recurrent = old_conv, old_rec
        for module, original in reversed(originals):
            module.forward = original


class State:
    """Save valid prefix KV and all GDN states; ignore uncomputed KV garbage."""
    def __init__(self, runner, seq, slot, length):
        self.length = length
        self.gdn = [(layer.conv_states[slot].clone(), layer.recurrent_states[slot].clone())
                    for layer in runner.gdn_layers]
        self.mapping = torch.tensor([seq.block_table[p//runner.block_size]*runner.block_size
            + p % runner.block_size for p in range(length)], device='cuda')
        flat = runner.kv_cache.reshape(*runner.kv_cache.shape[:2], -1, *runner.kv_cache.shape[-2:])
        self.kv = flat.index_select(2, self.mapping)

    def restore(self, runner, slot, *, gdn=True, kv=True, conv=True, recurrent=True):
        if gdn:
            for layer, (conv_state, rec) in zip(runner.gdn_layers, self.gdn):
                if conv:
                    layer.conv_states[slot].copy_(conv_state)
                if recurrent:
                    layer.recurrent_states[slot].copy_(rec)
        if kv:
            flat = runner.kv_cache.reshape(*runner.kv_cache.shape[:2], -1, *runner.kv_cache.shape[-2:])
            flat.index_copy_(2, self.mapping, self.kv)


def layer_trace(model):
    trace, hooks = {}, []
    for i, layer in enumerate(model.model.layers):
        def capture(module, inputs, output, i=i):
            hidden, residual = output
            # Keep both streams: their sum alone can hide compensating errors.
            value = torch.stack((hidden.reshape(-1, hidden.shape[-1]),
                                 residual.reshape(-1, residual.shape[-1])), dim=1).detach().cpu()
            trace.setdefault(i, []).append(value)
        hooks.append(layer.register_forward_hook(capture))
    return trace, hooks


@torch.inference_mode()
def experiment(args, prompt, name, mode):
    engine = LLMEngine(args.model, max_num_seqs=1, enforce_eager=True, enable_prefix_cache=False,
        max_model_len=len(prompt)+args.tokens+16, max_num_batched_tokens=max(2048, len(prompt)),
        gpu_memory_utilization=.6, speculative=SpeculativeConfig(enabled=True, verification_mode='packed'))
    runner = engine.model_runner
    hooks = []
    try:
        # Only allocation uses SpeculativeConfig: the scheduler never proposes tokens.
        engine.scheduler.speculative = None
        engine.add_request(prompt, SamplingParams(temperature=0, max_tokens=args.tokens+2, ignore_eos=True))
        engine.step()
        if engine.batch_queue or len(engine.scheduler.running) != 1:
            raise RuntimeError('Expected one fully consumed prefill')
        seq = engine.scheduler.running[0]
        if seq.num_cached_tokens != len(prompt) or seq.num_tokens != len(prompt)+1:
            raise RuntimeError('Prefill advanced beyond the shared initial state')
        slot = runner.input_batch.slots_for([seq])[0]
        if not engine.scheduler.block_manager.reserve_trial(seq, len(prompt)+args.tokens):
            raise RuntimeError('Cannot reserve diagnostic KV pages')
        runner.request_state.update([seq], [slot], [])
        reference = State(runner, seq, slot, len(prompt))
        candidate = reference
        trace, hooks = layer_trace(runner.model)
        rows, blocks, forced, norm_records, counterfactuals = [], [], [seq.last_token], [], []
        for offset in range(0, args.tokens, args.block):
            count = min(args.block, args.tokens-offset)
            position = len(prompt)+offset
            before = reference
            candidate_before = candidate
            reference.restore(runner, slot)
            trace.clear()
            logits = []
            inputs = []
            token = forced[-1]
            for i in range(count):
                inputs.append(token)
                physical = seq.block_table[(position+i)//runner.block_size]*runner.block_size+(position+i)%runner.block_size
                set_context(False, slot_mapping=torch.tensor([physical], device='cuda', dtype=torch.int32),
                    context_lens=torch.tensor([position+i+1], device='cuda', dtype=torch.int32),
                    block_tables=runner.request_state.block_tables.tensor[slot:slot+1],
                    state_indices=torch.tensor([slot], device='cuda'),
                    batch_descriptor=BatchDescriptor('spec_decode', 1, 1, 1, 1))
                hidden = runner.model(torch.tensor([token], device='cuda'), torch.tensor([position+i], device='cuda'))
                value = runner.model.compute_logits(hidden).reshape(-1).float().cpu()
                logits.append(value)
                token = int(value.argmax())
                forced.append(token)
            expected = torch.stack(logits)
            expected_trace = {i: torch.cat(values) for i, values in trace.items()}
            reference = State(runner, seq, slot, position+count)
            (before if mode == 'reset' else candidate).restore(runner, slot)
            trace.clear()
            # Capture final hidden directly; project one row at a time in BOTH paths.
            final = []
            audit = [] if offset in getattr(args, 'audit_norm_offsets', []) else None
            hook = runner.model.register_forward_hook(lambda m, x, y: final.append(y.detach()))
            try:
                plan = VerificationPlan(seq.seq_id, position, inputs[0], tuple(inputs[1:]))
                with intervention(runner.model, name, audit):
                    packed_forward(runner, seq, plan, slot, project=False)
                    actual = torch.cat([runner.model.compute_logits(row[None]).reshape(1, -1)
                        for row in final[0].reshape(count, -1)]).float().cpu()
            finally:
                hook.remove()
            candidate = State(runner, seq, slot, position+count)
            for entry in audit or []:
                operands = entry.pop('operands', None)
                entry.update(input_offset=offset, variant=name, state_mode=mode)
                if operands is not None:
                    directory = Path(args.json_out).with_suffix('').with_name(Path(args.json_out).stem+'_operands')
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory/f'{mode}_{name.replace(":", "_")}_{offset}_{entry["module"]}.pt'
                    torch.save(operands, path)
                    entry['operands_path'] = str(path)
                norm_records.append(entry)
            state_errors = [dict(layer=layer.layer_idx, conv=difference(a[0], b[0]),
                recurrent=difference(a[1], b[1])) for layer, a, b in
                zip(runner.gdn_layers, reference.gdn, candidate.gdn)]
            blocks.append(dict(input_offset=offset, query_count=count,
                states=state_errors, kv=difference(reference.kv, candidate.kv)))
            actual_trace = {i: torch.cat(values) for i, values in trace.items()}
            for i in range(count):
                winner = int(expected[i].argmax())
                contender = int(actual[i].argmax())
                top = expected[i].topk(2)
                margin = float(top.values[0]-top.values[1])
                other = contender if contender != winner else int(top.indices[1])
                delta = actual[i]-expected[i]
                layer_errors = [dict(layer=j, **difference(expected_trace[j][i], actual_trace[j][i]))
                                for j in expected_trace]
                rows.append(dict(input_position=position+i, output_token_number=offset+i+2,
                    reference_token=winner, candidate_token=contender, argmax_equal=winner == contender,
                    margin=margin, contender_reference_gap=float(expected[i,winner]-expected[i,other]),
                    directional_perturbation=float(delta[other]-delta[winner]),
                    candidate_gap=float(actual[i,winner]-actual[i,other]),
                    reference_logits_pair=[float(expected[i,winner]), float(expected[i,other])],
                    candidate_logits_pair=[float(actual[i,winner]), float(actual[i,other])],
                    logits=difference(expected[i], actual[i]), layers=layer_errors))
            if mode == 'rolling' and offset in getattr(args, 'counterfactual_offsets', []):
                trials = []
                for source in ('gdn', 'kv', 'both', 'conv', 'recurrent'):
                    candidate_before.restore(runner, slot)
                    before.restore(runner, slot, gdn=source in ('gdn', 'both', 'conv', 'recurrent'),
                                   kv=source in ('kv', 'both'), conv=source != 'recurrent',
                                   recurrent=source != 'conv')
                    final.clear()
                    hook = runner.model.register_forward_hook(lambda m, x, y: final.append(y.detach()))
                    try:
                        with intervention(runner.model, name):
                            packed_forward(runner, seq, plan, slot, project=False)
                            logits = torch.cat([runner.model.compute_logits(row[None]).reshape(1, -1)
                                for row in final[0].reshape(count, -1)]).float().cpu()
                    finally:
                        hook.remove()
                    predicted = logits.argmax(-1)
                    trials.append(dict(restored=source, logits=difference(expected, logits),
                        candidate_tokens=predicted.tolist(),
                        flip_output_tokens=[offset+i+2 for i in range(count)
                                            if predicted[i] != expected[i].argmax()]))
                counterfactuals.append(dict(input_offset=offset,
                    reference_tokens=expected.argmax(-1).tolist(),
                    rolling_tokens=actual.argmax(-1).tolist(),
                    rolling_logits=difference(expected, actual), trials=trials))
                # Counterfactual branches never alter the continuing rolling branch.
                candidate.restore(runner, slot)
            runner._trial_endpoints = {}
            reset_context()
        return dict(variant=name, state_mode=mode, block_size=args.block,
            teacher_tokens=forced, teacher_sha256=hashlib.sha256(json.dumps(forced).encode()).hexdigest(),
            first_flip=next((r['output_token_number'] for r in rows if not r['argmax_equal']), None),
            flips=sum(not r['argmax_equal'] for r in rows), rows=rows, blocks=blocks,
            norm_audit=norm_records, counterfactuals=counterfactuals)
    finally:
        for hook in hooks:
            hook.remove()
        reset_context()
        engine.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--fixtures', default='logs/validate/vllm_model_baseline_20260928.json')
    parser.add_argument('--case', default='repository_code')
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--block', type=int, default=5, help='Target query count, including anchor')
    parser.add_argument('--variants', nargs='+', default=['native', 'conv', 'bv', 'fma', 'recurrent',
        'gemm', 'conv+recurrent+gemm', 'conv+recurrent+gemm+attention'])
    parser.add_argument('--state-modes', nargs='+', choices=['reset', 'rolling'], default=['reset', 'rolling'])
    parser.add_argument('--json-out', default='logs/validate/target_numerics.json')
    parser.add_argument('--audit-norm-offsets', nargs='*', type=int, default=[])
    parser.add_argument('--counterfactual-offsets', nargs='*', type=int, default=[])
    args = parser.parse_args()
    if args.tokens < 1 or args.block < 1:
        parser.error('tokens and block must be positive')
    allowed = {'native', 'conv', 'bv', 'fma', 'recurrent', 'gemm', 'attention', 'pointwise',
               'gemma_norm', 'gated_norm', 'activation', 'norm_mean'}
    if any(any(flag not in allowed and not flag.startswith('norm:') for flag in name.split('+'))
           for name in args.variants):
        parser.error('Unknown intervention')
    fixtures = json.loads(Path(args.fixtures).read_text())
    prompt = next(row['prompt_ids'] for row in fixtures['cases']
                  if row['case'] == args.case and row['batch_size'] == 1)
    record = dict(completed=False, case=args.case, prompt_ids=prompt, model=args.model,
        dtype='bfloat16', eager=True, prefix_cache=False, tensor_parallel=1,
        logits_projection='single_row_both_paths', results=[], torch_version=torch.__version__)
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    for mode in args.state_modes:
        for name in args.variants:
            result = experiment(args, prompt, name, mode)
            record['results'].append(result)
            if len({r['teacher_sha256'] for r in record['results']}) != 1:
                raise AssertionError('Reference trajectory changed across experiments')
            output.write_text(json.dumps(record, indent=2))
            print(json.dumps({k: result[k] for k in ('variant', 'state_mode', 'first_flip', 'flips')}), flush=True)
    record['completed'] = True
    output.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
