"""Full-engine installed-vLLM reference using identical tokenized prompts.

Run baseline and MTP in separate processes so GPU model storage is released.
This supplements the pinned-source differential check; it does not claim that
the installed vLLM distribution is the pinned source revision.
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--mode', choices=('baseline', 'mtp'), required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--hybrid-reference', type=Path,
                        help='completed conservative HybridInfer fixtures; compare batch=1 tokens')
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--draft-tokens', type=int, default=4)
    parser.add_argument('--json-out', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'mtp' and args.baseline is None:
        parser.error('MTP comparison requires --baseline from the separate baseline process')
    if min(args.output_tokens, args.draft_tokens) < 1:
        parser.error('token counts must be positive')
    fixtures = json.loads(args.fixtures.read_text())
    cases = {}
    for row in fixtures['cases']:
        if not row['prefix_cache'] and row['case'] not in cases:
            cases[row['case']] = row['prompt_ids']
    if not cases:
        raise ValueError('fixtures contain no cold, prefix-disabled prompts')
    hybrid_refs = {}
    if args.hybrid_reference:
        hybrid = json.loads(args.hybrid_reference.read_text())
        if not hybrid['completed'] or not hybrid['passed']:
            raise ValueError('HybridInfer reference did not pass its own conservative gate')
        for row in hybrid['cases']:
            if not row['prefix_cache']:
                hybrid_refs.setdefault(row['case'], row)
    record = dict(completed=False, passed=False, mode=args.mode, model=args.model,
                  output_tokens=args.output_tokens, cases=[],
                  scope='installed-vLLM full engine; separate from pinned-source kernel alignment')
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import vllm
        from vllm import LLM, SamplingParams
        record.update(vllm_version=vllm.__version__, vllm_path=vllm.__file__,
                      torch_version=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(),
                      dtype='bfloat16', prefix_cache=False, enforce_eager=True,
                      mamba_ssm_cache_dtype='float32', max_num_batched_tokens=2048,
                      max_num_seqs=4)
        baseline = json.loads(args.baseline.read_text()) if args.baseline else None
        if baseline:
            if not baseline['completed'] or not baseline['execution_passed'] or baseline['mode'] != 'baseline':
                raise ValueError('baseline engine run was incomplete or failed execution')
            for field in ('model', 'output_tokens', 'vllm_version', 'torch_version', 'dtype',
                          'prefix_cache', 'enforce_eager', 'mamba_ssm_cache_dtype',
                          'max_num_batched_tokens', 'max_num_seqs'):
                if baseline[field] != record[field]:
                    raise ValueError(f'baseline configuration differs: {field}')
        llm = LLM(model=args.model, dtype='bfloat16', enforce_eager=True,
                  language_model_only=True, skip_tokenizer_init=True,
                  enable_prefix_caching=False, mamba_ssm_cache_dtype='float32',
                  gpu_memory_utilization=0.6, max_num_seqs=4,
                  max_model_len=max(map(len, cases.values()))+args.output_tokens+16,
                  max_num_batched_tokens=2048, seed=42,
                  speculative_config=None if args.mode == 'baseline' else
                  dict(method='mtp', num_speculative_tokens=args.draft_tokens))
        resolved = llm.llm_engine.vllm_config.speculative_config
        record['resolved_speculative_config'] = None if resolved is None else dict(
            method=resolved.method, num_speculative_tokens=resolved.num_speculative_tokens,
            draft_architectures=resolved.draft_model_config.architectures)
        if args.mode == 'mtp' and (resolved is None or resolved.method != 'mtp'
                                   or resolved.num_speculative_tokens != args.draft_tokens):
            raise AssertionError('vLLM did not activate the requested MTP backend')
        params = SamplingParams(temperature=0, max_tokens=args.output_tokens, ignore_eos=True)
        references = {(row['case'], row['batch_size']): row for row in baseline['cases']} if baseline else {}
        for batch_size in (1, 4):
            for name, ids in cases.items():
                prompts = [dict(prompt_token_ids=ids) for _ in range(batch_size)]
                outputs = llm.generate(prompts, params, use_tqdm=False)
                tokens = [list(output.outputs[0].token_ids) for output in outputs]
                lengths_ok = all(len(t) == args.output_tokens for t in tokens)
                same_prompt = all(list(output.prompt_token_ids) == ids for output in outputs)
                row = dict(case=name, batch_size=batch_size, prompt_ids=ids,
                           tokens=tokens, output_lengths_match=lengths_ok, prompt_ids_match=same_prompt)
                if baseline:
                    ref = references[(name, batch_size)]
                    if ref['prompt_ids'] != ids:
                        raise ValueError('baseline prompt IDs differ')
                    row['token_matches'] = [a == b for a, b in zip(tokens, ref['tokens'])]
                    row['first_mismatches'] = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
                                              for a, b in zip(tokens, ref['tokens'])]
                if batch_size == 1 and hybrid_refs:
                    ref = hybrid_refs[name]
                    if ref['prompt_ids'] != ids or len(ref['tokens']) < args.output_tokens:
                        raise ValueError('HybridInfer reference prompt/length differs')
                    expected = ref['tokens'][:args.output_tokens]
                    row['hybrid_token_match'] = tokens[0] == expected
                    row['hybrid_first_mismatch'] = next((i for i, (x, y) in enumerate(zip(tokens[0], expected)) if x != y), None)
                record['cases'].append(row)
                args.json_out.write_text(json.dumps(record, indent=2))
                print(json.dumps({k: v for k, v in row.items() if k not in ('tokens', 'prompt_ids')}), flush=True)
        record['completed'] = True
        record['execution_passed'] = all(row['output_lengths_match'] and row['prompt_ids_match'] for row in record['cases'])
        if hybrid_refs:
            record['hybrid_token_match'] = all(row.get('hybrid_token_match', True) for row in record['cases'])
        record['baseline_token_match'] = all(all(row.get('token_matches', [True])) for row in record['cases'])
        record['acceptance_scope'] = 'full_engine_execution; token matches are diagnostics'
        record['passed'] = record['execution_passed']
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        args.json_out.write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != 'cases'}, indent=2))
    if not record['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
