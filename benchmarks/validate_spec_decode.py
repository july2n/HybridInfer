"""Token, endpoint and state-recovery validation for speculative decoding.

Run with PYTHONPATH=src python benchmarks/validate_spec_decode.py --model models/Qwen3.5-0.8B
The same eager engine is reused at idle boundaries, keeping weights/kernels fixed.
Controlled drafts exercise every acceptance endpoint; these are validation-only
proposals, not evidence of n-gram acceptance rate or acceleration.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.spec_decode.ngram import NgramProposer
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.spec_decode.execution import packed_forward
from hybridinfer.utils.context import get_context


def snapshot(runner, seq):
    slot = runner.input_batch.slots_for([seq])[0]
    c = int(runner.request_state.computed.tensor[slot].item())
    pages = (c + runner.block_size - 1) // runner.block_size
    ids = torch.tensor(seq.block_table[:pages], dtype=torch.int64, device='cuda')
    kv = runner.kv_cache.index_select(2, ids)
    kv = kv.reshape(*kv.shape[:2], -1, *kv.shape[-2:])[:, :, :c].cpu().clone()
    return dict(computed=c,
                tokens=runner.request_state.tokens.tensor[slot, :c+1].cpu().clone(), kv=kv,
                conv=[layer.conv_states[slot].cpu().clone() for layer in runner.gdn_layers],
                recurrent=[layer.recurrent_states[slot].cpu().clone() for layer in runner.gdn_layers])


def equal_state(reference, trial):
    failures = []
    for key in ('tokens', 'kv'):
        if not torch.equal(reference[key], trial[key]):
            failures.append(key)
    for key in ('conv', 'recurrent'):
        if not all(torch.equal(a, b) for a, b in zip(reference[key], trial[key])):
            failures.append(key)
    return failures


def state_drift(reference, trial):
    diagnostics = {}
    for key in ('kv', 'conv', 'recurrent'):
        pairs = [(reference[key], trial[key])] if key == 'kv' else zip(reference[key], trial[key])
        max_abs, square_sum, count, finite = 0., 0., 0, True
        for before, after in pairs:
            finite = finite and bool(torch.isfinite(after).all())
            delta = after.float()-before.float()
            max_abs = max(max_abs, float(delta.abs().max()))
            square_sum += float(delta.square().sum())
            count += delta.numel()
        diagnostics[key] = dict(max_abs=max_abs, rmse=(square_sum/max(1, count))**0.5, finite=finite)
    return diagnostics


def same_block_reference(runner, seq, plan, result, before):
    private_slot = runner.config.max_num_seqs
    for layer, conv, recurrent in zip(runner.gdn_layers, before['conv'], before['recurrent']):
        layer.conv_states[private_slot].copy_(conv)
        layer.recurrent_states[private_slot].copy_(recurrent)
    count = result.committed_computed_length-plan.computed_length
    replay = VerificationPlan(plan.request_id, plan.computed_length, plan.anchor, plan.candidates[:count-1])
    packed_forward(runner, seq, replay, private_slot, project=False)
    ref = snapshot(runner, seq)
    ref['conv'] = [layer.conv_states[private_slot].cpu().clone() for layer in runner.gdn_layers]
    ref['recurrent'] = [layer.recurrent_states[private_slot].cpu().clone() for layer in runner.gdn_layers]
    return ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--json-out', default='logs/validate/spec_decode.json')
    parser.add_argument('--verification-mode', choices=('packed', 'sequential', 'packed_guarded'),
                        default=SpeculativeConfig().verification_mode)
    args = parser.parse_args()
    engine = LLMEngine(args.model, max_num_seqs=2, max_model_len=1024,
                       max_num_batched_tokens=1024, gpu_memory_utilization=0.6,
                       enforce_eager=True, enable_prefix_cache=True, speculative=SpeculativeConfig(enabled=True, verification_mode=args.verification_mode))
    records = []
    runner = engine.model_runner
    original_sample = runner.sample_tokens
    original_verify = runner.verify_speculative
    original_propose = NgramProposer._propose_one
    original_logits = runner.model.compute_logits
    normal_eos = engine.config.eos
    checkpoints = engine.scheduler.checkpoints
    try:
        for prefix in (False, True):
            engine.scheduler.enable_prefix_cache = prefix
            engine.scheduler.checkpoints = checkpoints if prefix else None
            for termination in ('length', 'eos', 'context'):
                # Exercise a page crossing in the decode window.
                prompt = engine.tokenizer.encode('Repeat this sequence: alpha beta alpha beta alpha beta.\n')
                prompt = (prompt * (509 // len(prompt) + 1))[:509]
                engine.config.max_model_len = engine.scheduler.max_model_len = 513 if termination == 'context' else 1024
                engine.config.eos = engine.scheduler.eos = 999 if termination == 'eos' else normal_eos
                def terminal_logits(hidden):
                    logits = original_logits(hidden)
                    context = get_context()
                    if termination == 'eos':
                        if not context.is_prefill:
                            endpoint = int(context.context_lens[0].item())
                            eos_row = 0 if endpoint == len(prompt)+2 else None
                        elif context.batch_descriptor.mode == 'spec_decode':
                            start = int(context.cu_seqlens_k[-1].item()) - len(logits)
                            eos_row = len(prompt)+1-start
                            if not 0 <= eos_row < len(logits):
                                eos_row = None
                        else:
                            eos_row = None
                        if eos_row is not None:
                            logits[eos_row].fill_(-100.)
                            logits[eos_row, 999] = 100.
                    return logits
                runner.model.compute_logits = terminal_logits
                engine.scheduler.speculative = None
                reference = {}
                def capture_sample():
                    seqs = runner._pending[2]
                    result = original_sample()
                    for seq in seqs:
                        data = snapshot(runner, seq)
                        reference[data['computed']] = data
                    return result
                runner.sample_tokens = capture_sample
                params = SamplingParams(temperature=0, max_tokens=12, ignore_eos=termination != 'eos')
                baseline = engine.generate([prompt], params, use_tqdm=False)[0]['token_ids']
                runner.sample_tokens = original_sample
                for accepted in (*range(5), None):
                    checks, errors = [], []
                    engine.scheduler.speculative = SpeculativeConfig(enabled=True, max_draft_tokens=4, verification_mode=args.verification_mode)
                    if accepted is not None:
                        def controlled(self, context, accepted=accepted):
                            length = len(context.token_ids) - len(prompt)
                            k = min(4, context.remaining_output_tokens-1,
                                    context.max_model_len-context.computed_length-1,
                                    context.verification_budget-1)
                            draft = list(baseline[length:length+k])
                            if accepted < len(draft):
                                draft[accepted] = (draft[accepted]+1) % engine.config.hf_config.vocab_size
                            return tuple(draft)
                        NgramProposer._propose_one = controlled
                    else:
                        NgramProposer._propose_one = original_propose
                    def capture_verify(seq, plan):
                        before_state = snapshot(runner, seq) if args.verification_mode == 'packed' else None
                        result = original_verify(seq, plan)
                        data = snapshot(runner, seq)
                        base = reference[data['computed']]
                        drift = state_drift(base, data)
                        if args.verification_mode == 'packed':
                            ref = same_block_reference(runner, seq, plan, result, before_state)
                            mismatch = equal_state(ref, data)
                            if not torch.equal(base['tokens'], data['tokens']):
                                mismatch.append('baseline_tokens')
                        else:
                            mismatch = equal_state(base, data)
                        if not all(value['finite'] for value in drift.values()):
                            mismatch.append('nonfinite_state')
                        errors.extend(mismatch)
                        checks.append(dict(computed=data['computed'], accepted=result.accepted_draft_tokens,
                                           output=result.output_length, mismatches=mismatch, baseline_state_drift=drift))
                        return result
                    runner.verify_speculative = capture_verify
                    before = dict(runner.spec_metrics)
                    prefix_hits_before = checkpoints.hits
                    start = perf_counter()
                    actual = engine.generate([prompt], params, use_tqdm=False)[0]['token_ids']
                    elapsed = perf_counter()-start
                    metrics = {key: value-before.get(key, 0) for key, value in runner.spec_metrics.items()}
                    record = dict(termination=termination, prefix_cache_hits=checkpoints.hits-prefix_hits_before, prefix_cache=prefix, forced_acceptance=accepted, token_match=actual == baseline,
                                  state_checks=checks, state_match=not errors, metrics=metrics,
                                  diagnostic_elapsed_seconds=elapsed)
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    if (actual != baseline or errors or (accepted is not None and not checks)
                            or (prefix and checkpoints.hits == prefix_hits_before)):
                        raise AssertionError('speculative reference disagrees with baseline')
                runner.verify_speculative = original_verify
    finally:
        runner.sample_tokens = original_sample
        runner.verify_speculative = original_verify
        NgramProposer._propose_one = original_propose
        runner.model.compute_logits = original_logits
        from hybridinfer.utils.context import reset_context
        reset_context()
        engine.exit()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(torch=torch.__version__, model=args.model,
                                       path=args.verification_mode+'_eager_reference', cases=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
