"""Strict CUDA norm comparison with installed vLLM's compiled static functions."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import torch
from hybridinfer.layers.norm_kernels import gemma_norm


def load_reference():
    path = Path(importlib.util.find_spec('vllm').origin).parent/'model_executor/layers/layernorm.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GemmaRMSNorm')
    namespace = dict(torch=torch)
    methods = ['_forward_static_no_residual', '_forward_static_with_residual']
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    if len(nodes) != len(methods):
        raise ValueError('Installed vLLM static norm contract changed')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return [namespace[name].__func__ for name in methods], dict(
        version=importlib.metadata.version('vllm'), source=str(path),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        torch_version=torch.__version__, precision_casts=False, dynamic=False,
        tests_full_engine=False)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-out', default='logs/validate/target_norm_kernel.json')
    args = parser.parse_args()
    functions, reference = load_reference()
    record = dict(completed=False, passed=False, reference=reference, cases=[])
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for residual_enabled, function in enumerate(functions):
            compiled = torch.compile(function, dynamic=False,
                                     options={'emulate_precision_casts': False})
            for width in (256, 1024):
                torch.manual_seed(42+width)
                weight = torch.randn(width, device='cuda', dtype=torch.bfloat16)*.2
                for count in (1, 5, 17):
                    # Strided input covers the query/gate projection view.
                    storage = torch.randn(count, 2*width, device='cuda', dtype=torch.bfloat16)
                    x = storage[:, :width]
                    residual = torch.randn_like(x) if residual_enabled else None
                    args_ref = (weight, 1e-6, x) if residual is None else (weight, 1e-6, x, residual)
                    expected = compiled(*args_ref)
                    actual = gemma_norm(x, weight, 1e-6, residual)
                    expected = expected if isinstance(expected, tuple) else (expected,)
                    actual = actual if isinstance(actual, tuple) else (actual,)
                    split = [gemma_norm(x[i:i+1], weight, 1e-6,
                                        None if residual is None else residual[i:i+1])
                             for i in range(count)]
                    split = [v if isinstance(v, tuple) else (v,) for v in split]
                    split = tuple(torch.cat([v[j] for v in split]) for j in range(len(actual)))
                    row = dict(width=width, count=count, residual=bool(residual_enabled),
                        reference_equal=all(torch.equal(a, b) for a, b in zip(actual, expected)),
                        rowwise_equal=all(torch.equal(a, b) for a, b in zip(actual, split)),
                        max_abs=max(float((a.float()-b.float()).abs().max())
                                    for a, b in zip(actual, expected)))
                    record['cases'].append(row)
                    print(json.dumps(row), flush=True)
        saved_path = Path('logs/validate/target_norm_audit_operands/'
            'reset_conv+recurrent+gemm+attention_90_model.layers.9.post_attention_layernorm.pt')
        if saved_path.exists():
            saved = torch.load(saved_path, weights_only=True)
            x, residual, weight = (saved[k].cuda() for k in ('x', 'residual', 'weight'))
            expected = compiled(weight, saved['eps'], x, residual)
            actual = gemma_norm(x, weight, saved['eps'], residual)
            record['captured_checkpoint_equal'] = all(torch.equal(a, b) for a, b in zip(actual, expected))
        record['completed'] = True
        record['passed'] = (all(r['reference_equal'] and r['rowwise_equal'] for r in record['cases'])
                            and record.get('captured_checkpoint_equal', True))
    finally:
        output.write_text(json.dumps(record, indent=2)+'\n')
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
