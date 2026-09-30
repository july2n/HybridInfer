"""Real MTP protocol/endpoints; cross-path generation differences are diagnostics."""
import argparse
import ast
import hashlib
from types import SimpleNamespace
import json
from pathlib import Path
import torch
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.engine.spec_verification import batch_execution
from spec_validation import check_original_endpoints


@torch.inference_mode()
def check_forward_contract(runner):
    """Execute pinned vLLM's unchanged MTP forward with shared local operators.

    This checks feature/norm/concatenation/layer order, not vLLM kernel numerics.
    Full-engine token comparisons are reported independently of this contract.
    """
    import subprocess
    from validate_vllm_spec_alignment import PINNED_COMMIT
    from hybridinfer.utils.context import set_context, reset_context
    root = Path('/home/lang/workspace/vllm')
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Unexpected MTP reference revision')
    path = root/'vllm/model_executor/models/qwen3_5_mtp.py'
    if subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', str(path)]).returncode:
        raise ValueError('Modified reference MTP source')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5MultiTokenPredictor')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    ns = dict(torch=torch, IntermediateTensors=dict,
              get_pp_group=lambda: SimpleNamespace(is_last_rank=True))
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(path), 'exec'), ns)
    proposer = runner.draft_proposer
    model = proposer.model
    model.layers[0].use_attn_reduce_scatter_for_moe = False
    reference = SimpleNamespace(num_mtp_layers=1, fc=model.fc,
        pre_fc_norm_hidden=model.pre_fc_norm_hidden,
        pre_fc_norm_embedding=model.pre_fc_norm_embedding, norm=model.norm, layers=model.layers)
    device = runner.model.lm_head.weight.device
    ids = torch.tensor([1, 2, 3], device=device)
    positions = torch.arange(3, device=device)
    features = torch.arange(3*runner.config.hf_config.hidden_size, device=device,
                            dtype=torch.float32).reshape(3, -1).sin().to(runner.model.lm_head.weight.dtype)
    embedding = runner.model.model.embed_tokens(ids)
    def context():
        set_context(True, slot_mapping=torch.arange(3, device=device, dtype=torch.int32),
            block_tables=torch.arange(proposer.pages, device=device, dtype=torch.int32)[None],
            cu_seqlens_q=torch.tensor([0, 3], device=device, dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 3], device=device, dtype=torch.int32), max_seqlen_q=3, max_seqlen_k=3)
    try:
        context()
        actual = model(embedding, features, positions)
        context()
        expected = ns['forward'](reference, ids, positions, features, None, embedding, 0)
        if not torch.equal(actual, expected):
            raise AssertionError('MTP forward differs from pinned vLLM operator order')
        return dict(passed=True, revision=PINNED_COMMIT, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    shared_local_operators=True)
    finally:
        reset_context()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--method', choices=['mtp', 'eagle3', 'dflash', 'dspark'], default='mtp')
    parser.add_argument('--draft-model', help='Local trained draft checkpoint directory')
    parser.add_argument('--vllm-baseline', help='Optional completed full-engine reference JSON')
    parser.add_argument('--json-out', default='logs/validate/mtp_alignment_implementation.json')
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--draft-tokens', type=int, default=1)
    parser.add_argument('--draft-sampling', choices=['greedy', 'random'], default='greedy')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt-tokens', type=int, default=509)
    parser.add_argument('--check-vllm-forward', action='store_true', help='Check pinned source operator-order contract')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--modes', nargs='+', choices=['baseline', 'packed', 'packed_guarded'],
                        default=['baseline', 'packed', 'packed_guarded'])
    parser.add_argument('--gpu-memory-utilization', type=float, default=.6)
    parser.add_argument('--state-snapshot-budget-mb', type=int, default=256)
    args = parser.parse_args()
    if args.method in ('eagle3', 'dflash', 'dspark') and (not args.draft_model or args.check_vllm_forward):
        parser.error('This backend requires --draft-model; --check-vllm-forward is MTP-specific')
    if args.draft_sampling == 'random' and any(m not in ('baseline', 'packed') for m in args.modes):
        parser.error('random drafts require --modes baseline packed (or packed)')
    if min(args.output_tokens, args.prompt_tokens, args.draft_tokens, *args.batch_sizes) < 1:
        parser.error('lengths and batch sizes must be positive')
    if args.vllm_baseline:
        source = json.loads(Path(args.vllm_baseline).read_text())
        if not source.get('completed') or not source.get('execution_passed'):
            raise ValueError('vLLM baseline did not complete its structural checks')
        if args.output_tokens > source['output_tokens']:
            raise ValueError('Requested output length exceeds available vLLM baseline')
        if Path(args.model).resolve() != Path(source['model']).resolve():
            raise ValueError('MTP model path differs from the vLLM baseline')
        fixtures = {row['case']: row['prompt_ids'] for row in source['cases'] if row['batch_size'] == 1}
        references = {(row['case'], row['batch_size']): row for row in source['cases']}
    else:
        from transformers import AutoTokenizer
        from spec_workloads import natural_cases, prepare_prompt
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        fixtures = {name: prepare_prompt(tokenizer, text, args.prompt_tokens, 'truncate')
                    for name, text in natural_cases().items()}
        references = {}
    record = dict(acceptance_scope='execution_and_original_endpoint_selection',
                  completed=False, passed=False, cases=[], endpoint_checks=0, endpoint_failures=[],
                  output_tokens=args.output_tokens, max_draft_tokens=args.draft_tokens, reference=args.vllm_baseline,
                  draft_sampling=args.draft_sampling, temperature=args.temperature, seed=args.seed,
                  method=args.method, draft_model=args.draft_model)
    own_references = {}
    capture = {}
    original_forward = batch_execution.packed_batch_forward

    def forward(*a, **kw):
        result = original_forward(*a, **kw)
        capture['states'] = a[0]._trial_endpoints
        return result
    batch_execution.packed_batch_forward = forward
    try:
        for mode in args.modes:
            spec = None if mode == 'baseline' else SpeculativeConfig(
                enabled=True, method=args.method, draft_model=args.draft_model,
                max_draft_tokens=args.draft_tokens, verification_mode=mode,
                state_snapshot_budget_mb=args.state_snapshot_budget_mb,
                mtp_draft_sampling=args.draft_sampling)
            engine = LLMEngine(args.model, max_num_seqs=max(args.batch_sizes),
                max_model_len=max(map(len, fixtures.values()))+args.output_tokens+16,
                max_num_batched_tokens=2048, gpu_memory_utilization=args.gpu_memory_utilization,
                enforce_eager=True, enable_prefix_cache=False, speculative=spec)
            runner = engine.model_runner
            if spec:
                record['weight_report'] = runner.draft_proposer.weight_report
                if args.check_vllm_forward:
                    record['mtp_forward_contract'] = check_forward_contract(runner)
                original_verify = runner.verify_speculative_batch
                def verify(seqs, plans):
                    capture.clear()
                    if args.draft_sampling == 'random':
                        for plan in plans:
                            q = plan.draft_probabilities
                            if q is None or q.shape != (len(plan.candidates), runner.config.hf_config.vocab_size):
                                raise AssertionError('Missing or misaligned realized MTP q')
                            if not torch.isfinite(q).all() or (q < 0).any():
                                raise AssertionError('Invalid MTP probability values')
                            torch.testing.assert_close(q.sum(-1), torch.ones(q.shape[0], device=q.device))
                            if plan.candidates:
                                tokens = torch.tensor(plan.candidates, device=q.device)
                                if not (q.gather(1, tokens[:, None]) > 0).all():
                                    raise AssertionError('Sampled MTP candidate outside q support')
                            record['draft_probability_checks'] = record.get('draft_probability_checks', 0)+1
                    handle = original_verify(seqs, plans)
                    results = handle.get_output()
                    if mode == 'packed' and capture.get('states'):
                        failures = check_original_endpoints(runner, seqs, plans, results, capture['states'])
                        record['endpoint_checks'] += len(results)
                        record['endpoint_failures'].extend(failures)
                    for seq, plan, result in zip(seqs, plans, results):
                        slot = runner.input_batch.seq_id_to_slot[seq.seq_id]
                        end = result.committed_computed_length
                        state = runner.request_state
                        start = plan.computed_length+1
                        if (int(state.computed.tensor[slot]) != end
                            or end != plan.computed_length+result.output_length
                            or state.tokens.tensor[slot, start:end+1].tolist() != list(result.token_ids)):
                            record['endpoint_failures'].append(dict(request=seq.seq_id, failed=['history/endpoint']))
                    capture.clear()
                    return handle
                runner.verify_speculative_batch = verify
            try:
                for batch_size in args.batch_sizes:
                    for name, prompt in fixtures.items():
                        output = engine.generate([prompt]*batch_size,
                            SamplingParams(temperature=args.temperature, seed=args.seed,
                                           max_tokens=args.output_tokens, ignore_eos=True), use_tqdm=False)
                        tokens = [row['token_ids'] for row in output]
                        key = (name, batch_size)
                        if mode == 'baseline':
                            own_references[key] = tokens
                        own = own_references.get(key)
                        external = references.get(key)
                        def compare(ref):
                            if ref is None:
                                return None
                            return [next((i for i, (x, y) in enumerate(zip(a, b[:args.output_tokens])) if x != y),
                                         None if len(a) == args.output_tokens else len(a))
                                    for a, b in zip(tokens, ref)]
                        row = dict(mode=mode, case=name, batch_size=batch_size, tokens=tokens,
                            lengths_ok=all(len(t) == args.output_tokens for t in tokens),
                            hybrid_first_mismatches=compare(own),
                            vllm_first_mismatches=compare(external['tokens'] if external else None))
                        record['cases'].append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != 'tokens'}), flush=True)
                        if spec and (runner.draft_proposer.last_hidden or any(x is not None for x in runner.draft_proposer.owners)):
                            raise AssertionError('Finished request retained MTP cache ownership')
                if spec:
                    record.setdefault('metrics', {})[mode] = dict(runner.spec_metrics)
                    # Exercise mixed greedy/random target requests through one batch.
                    if mode == 'packed':
                        sampled = engine.generate([next(iter(fixtures.values()))]*2,
                            [SamplingParams(temperature=t, seed=args.seed+i, max_tokens=32, ignore_eos=True)
                             for i, t in enumerate([0, .8])], use_tqdm=False)
                        record['random_lengths'] = [len(x['token_ids']) for x in sampled]
                        if args.draft_sampling == 'random':
                            repeated = engine.generate([next(iter(fixtures.values()))]*2,
                                [SamplingParams(temperature=t, seed=args.seed+i, max_tokens=32, ignore_eos=True)
                                 for i, t in enumerate([0, .8])], use_tqdm=False)
                            record['seed_replay_match'] = all(a['token_ids'] == b['token_ids']
                                for a, b in zip(sampled, repeated))
                            if not record['seed_replay_match']:
                                raise AssertionError('Explicit MTP seeds did not reproduce mixed generation')
            finally:
                engine.exit()
        record['completed'] = True
        speculative_modes = [m for m in args.modes if m != 'baseline']
        record['speculative_rounds_ok'] = all(
            record.get('metrics', {}).get(m, {}).get('rounds', 0) > 0
            and record['metrics'][m].get('draft_tokens', 0) > 0 for m in speculative_modes)
        record['packed_endpoint_coverage_ok'] = ('packed' not in speculative_modes
            or record['endpoint_checks'] > 0)
        record['execution_passed'] = (record['speculative_rounds_ok']
            and record['packed_endpoint_coverage_ok']
            and not record['endpoint_failures']
            and all(x['lengths_ok'] for x in record['cases'])
            and record.get('random_lengths', [32, 32]) == [32, 32])
        record['hybrid_token_match'] = all(x['hybrid_first_mismatches'] is not None
            and all(i is None for i in x['hybrid_first_mismatches']) for x in record['cases'])
        record['vllm_token_match'] = None if not references else all(x['vllm_first_mismatches'] is not None
            and all(i is None for i in x['vllm_first_mismatches']) for x in record['cases'])
        record['passed'] = record['execution_passed']
    finally:
        batch_execution.packed_batch_forward = original_forward
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('cases', 'weight_report')}, indent=2))
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
