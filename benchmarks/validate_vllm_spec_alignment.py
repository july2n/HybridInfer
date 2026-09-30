"""Execute pinned vLLM source functions against HybridInfer's contracts.

This is a kernel/metadata differential check, not an end-to-end vLLM run.
Only selected definitions are loaded, with their original AST and filename;
vLLM package initialization and compiled extensions are not needed. Imports
are supplied explicitly. No reference algorithm is reimplemented here.
"""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from hybridinfer.sampling.batch_verifier import accept_greedy_batch
from hybridinfer.spec_decode.interfaces import VerificationPlan
from hybridinfer.spec_decode.metadata import VerificationBatch


PINNED_COMMIT = 'a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb'


def load_definitions(path, namespace, names=None, assignments=False):
    """Compile unchanged definitions; retain source locations for Triton JIT."""
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if names is None or node.name in names:
                definitions.append(node)
        elif assignments and isinstance(node, (ast.Assign, ast.AnnAssign)):
            definitions.append(node)
    if names is not None:
        found = {node.name for node in definitions if hasattr(node, 'name')}
        if found != set(names):
            raise ValueError(f'missing source definitions in {path}: {set(names)-found}')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)


def load_reference(root):
    files = {
        'metadata': 'vllm/v1/spec_decode/metadata.py',
        'runner': 'vllm/v1/worker/gpu_model_runner.py',
        'legacy_sampler': 'vllm/v1/sample/rejection_sampler.py',
        'gumbel': 'vllm/v1/worker/gpu/sample/gumbel.py',
        'watermark': 'vllm/v1/worker/gpu/sample/watermark.py',
        'philox': 'vllm/v1/watermarking/prfs/philox.py',
        'gpu_sampler': 'vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py',
    }
    paths = {key: root / value for key, value in files.items()}
    clean = subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', *files.values()])
    if clean.returncode:
        raise ValueError('reference source differs from the pinned commit')
    base = dict(torch=torch, np=np, triton=triton, tl=tl,
                tldevice=libdevice, dataclass=dataclass, __name__=__name__)
    metadata_ns = dict(base)
    load_definitions(paths['metadata'], metadata_ns)
    metadata_ns['async_tensor_h2d'] = lambda values, device: torch.as_tensor(values, device=device)
    runner_tree = ast.parse(paths['runner'].read_text(), filename=str(paths['runner']))
    runner_class = next(n for n in runner_tree.body if isinstance(n, ast.ClassDef)
                        and n.name == 'GPUModelRunner')
    methods = [n for n in runner_class.body if isinstance(n, ast.FunctionDef)
               and n.name in ('_get_cumsum_and_arange', '_calc_spec_decode_metadata')]
    if len(methods) != 2:
        raise ValueError('pinned runner metadata methods missing')
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(paths['runner']), 'exec'), metadata_ns)

    legacy_ns = dict(base, PLACEHOLDER_TOKEN_ID=tl.constexpr(-1))
    load_definitions(paths['legacy_sampler'], legacy_ns, ['rejection_greedy_sample_kernel'])
    gpu_ns = dict(base)
    load_definitions(paths['gumbel'], gpu_ns, assignments=True)
    philox_tree = ast.parse(paths['philox'].read_text())
    constants = [node for node in philox_tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id.startswith('_')
                 and isinstance(node.value, (ast.Constant, ast.BinOp))]
    constant_ns = {}
    exec(compile(ast.Module(body=constants, type_ignores=[]), str(paths['philox']), 'exec'), constant_ns)
    gpu_ns.update({name+'_VALUE': value for name, value in constant_ns.items() if name.startswith('_')})
    gpu_ns.update(HAS_TRITON=True, tl_math=tl.math)
    load_definitions(paths['watermark'], gpu_ns, assignments=True)
    load_definitions(paths['gpu_sampler'], gpu_ns)
    hashes = {files[key]: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    return metadata_ns, legacy_ns['rejection_greedy_sample_kernel'], gpu_ns['rejection_sample'], hashes


def compare_metadata(ns, rng, device, trials):
    checked = 0
    layouts = [([3, 0, 2], [4, 100, 3]), ([0, 1, 4], [17, 2, 5]),
               ([0, 0, 0], [1, 20, 1]), ([3, 0, 2, 0, 1], [4, 100, 3, 100, 2])]
    for _ in range(trials):
        k = rng.integers(0, 9, size=int(rng.integers(1, 17))).tolist()
        layouts.append((k, (np.asarray(k)+1+rng.integers(0, 33, size=len(k))).tolist()))
    for counts, scheduled in layouts:
        plans = tuple(VerificationPlan(i, 31+i, 100+i, tuple(rng.integers(0, 97, size=k).tolist()))
                      for i, k in enumerate(counts))
        batch = VerificationBatch(plans, tuple(scheduled))
        ours = batch.tensors(device)
        inputs = []
        for plan, count in zip(plans, scheduled):
            inputs.extend([999]*(count-len(plan.input_tokens)))
            inputs.extend(plan.input_tokens)
        runner = SimpleNamespace(device=device, arange_np=np.arange(len(inputs)+1),
                                 _arange_scratch=np.empty(len(inputs)+1, dtype=np.int32),
                                 input_ids=SimpleNamespace(gpu=torch.tensor(inputs, device=device)))
        runner._get_cumsum_and_arange = lambda *a, **kw: ns['_get_cumsum_and_arange'](runner, *a, **kw)
        ref = ns['_calc_spec_decode_metadata'](runner, np.asarray(counts, dtype=np.int32),
                                              np.cumsum(scheduled, dtype=np.int32))
        for field in ('logits_indices', 'target_logits_indices', 'bonus_logits_indices',
                      'cu_num_draft_tokens', 'cu_num_sampled_tokens', 'draft_token_ids'):
            if not torch.equal(getattr(ours, field).long(), getattr(ref, field).long()):
                raise AssertionError(f'{field}: counts={counts}, scheduled={scheduled}')
        checked += 1
    return checked


def compare_greedy(legacy, gpu_sample, rng, trials):
    checked = requests = 0
    layouts = [[3, 0, 2], [0, 1, 4], [0, 0, 0]]
    layouts += [[k]*(k+1) for k in range(9)]
    layouts += [rng.integers(0, 9, size=int(rng.integers(1, 17))).tolist() for _ in range(trials)]
    for trial, counts in enumerate(layouts):
        plans, predictions, logits_rows = [], [], []
        max_k = max(counts)
        vocab = 17 if trial % 2 else 1031
        for i, k in enumerate(counts):
            candidates = rng.integers(0, vocab, size=k).tolist()
            accepted = i if trial in range(3, 12) else int(rng.integers(0, k+1))
            targets = list(candidates) + [int(rng.integers(0, vocab))]
            if accepted < k:
                targets[accepted] = (candidates[accepted]+1) % vocab
                # Later rows deliberately match rejected candidates.
            plans.append(VerificationPlan(i, 50+i, 7, tuple(candidates)))
            predictions.extend(targets)
            for winner in targets:
                row = torch.full((vocab,), -4., dtype=torch.bfloat16, device='cuda')
                row[winner] = 2.
                # Exercise BF16 ties while retaining PyTorch's smallest-id argmax.
                if trial % 3 == 0 and winner+1 < vocab:
                    row[winner+1] = 2.
                logits_rows.append(row)
        batch = VerificationBatch.from_plans(plans)
        meta = batch.tensors('cuda')
        predictions_t = torch.tensor(predictions, device='cuda')
        ours = accept_greedy_batch(batch, meta, predictions_t,
                                  remaining_output_tokens=[100]*len(plans), max_model_len=1000)
        old_out = torch.full_like(ours.token_ids, -1)
        legacy[(len(plans),)](old_out, meta.cu_num_draft_tokens.int(), meta.draft_token_ids.int(),
                             predictions_t[meta.target_logits_indices],
                             predictions_t[meta.bonus_logits_indices], None, max_k,
                             None, None, SYNTHETIC_MODE=False)
        if not torch.equal(old_out, ours.token_ids):
            raise AssertionError(f'legacy greedy mismatch: {counts}')
        cu = torch.cat((torch.zeros(1, device='cuda', dtype=torch.int32),
                        meta.cu_num_sampled_tokens.int()))
        idx = torch.tensor(rng.permutation(len(plans)), device='cuda', dtype=torch.int32)
        expanded_idx = torch.repeat_interleave(idx, torch.tensor(counts, device='cuda')+1)
        local_pos = torch.tensor([j for k in counts for j in range(k+1)], device='cuda', dtype=torch.int32)
        positions = local_pos.long()+50
        draft_inputs = torch.tensor([t for p in plans for t in p.input_tokens], device='cuda')
        new_out, new_lengths = gpu_sample(torch.stack(logits_rows), None, draft_inputs,
                                         cu, positions, idx, expanded_idx, local_pos,
                                         torch.zeros(len(plans), device='cuda'),
                                         torch.arange(len(plans), device='cuda', dtype=torch.int64), max_k)
        # New runner does not initialize unused output slots. Compare valid tokens only.
        valid = torch.arange(max_k+1, device='cuda')[None, :] < new_lengths[:, None]
        if not torch.equal(new_lengths.long(), ours.lengths) or not torch.equal(new_out[valid], ours.token_ids[valid]):
            raise AssertionError(f'GPU runner greedy mismatch: {counts}')
        # EOS/length clipping is an output-consumer contract, after vLLM rejection.
        remaining = [int(rng.integers(1, max_k+2)) for _ in plans]
        ignored = rng.integers(0, 2, size=len(plans)).astype(bool).tolist()
        eos = int(predictions_t[meta.bonus_logits_indices[0]])
        clipped = accept_greedy_batch(batch, meta, predictions_t,
                                     remaining_output_tokens=remaining, max_model_len=1000,
                                     eos=eos, ignore_eos=ignored)
        for row, plan in enumerate(plans):
            emitted = old_out[row, :int(ours.lengths[row])].tolist()[:remaining[row]]
            if not ignored[row] and eos in emitted:
                emitted = emitted[:emitted.index(eos)+1]
            if clipped.token_ids[row, :int(clipped.lengths[row])].tolist() != emitted:
                raise AssertionError('termination clipping mismatch')
            if int(clipped.computed[row]) != plan.computed_length+len(emitted):
                raise AssertionError('committed endpoint mismatch')
        checked += 1
        requests += len(plans)
    return dict(batches=checked, requests=requests, legacy_greedy=True,
                gpu_runner_greedy=True, eos_and_output_limits=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-root', type=Path, default=Path('/home/lang/workspace/vllm'))
    parser.add_argument('--expected-commit', default=PINNED_COMMIT)
    parser.add_argument('--trials', type=int, default=100)
    parser.add_argument('--json-out', type=Path, default=Path('logs/validate/vllm_spec_alignment.json'))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    commit = subprocess.check_output(['git', '-C', str(args.vllm_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != args.expected_commit:
        raise ValueError(f'vLLM revision changed: expected {args.expected_commit}, got {commit}')
    record = dict(completed=False, passed=False, vllm_commit=commit,
                  scope='source-function differential, not full-engine/model compatibility',
                  torch_version=torch.__version__, triton_version=triton.__version__, seed=42)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ns, legacy, gpu_sample, hashes = load_reference(args.vllm_root)
        record['source_sha256'] = hashes
        rng = np.random.default_rng(42)
        record['metadata_cpu_layouts'] = compare_metadata(ns, rng, 'cpu', args.trials)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; a CPU-only run cannot pass this gate')
        record['gpu'] = torch.cuda.get_device_name()
        record['metadata_cuda_layouts'] = compare_metadata(ns, rng, 'cuda', args.trials)
        record['acceptance'] = compare_greedy(legacy, gpu_sample, rng, args.trials)
        record['completed'] = record['passed'] = True
    except Exception as exc:
        record['error'] = repr(exc)
        raise
    finally:
        args.json_out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
