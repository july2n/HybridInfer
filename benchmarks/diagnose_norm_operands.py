"""Replay captured GemmaRMSNorm operands and compare each eager CUDA stage."""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import hashlib
import subprocess
from pathlib import Path

import torch
from diagnose_target_numerics import difference
from validate_vllm_spec_alignment import load_definitions, PINNED_COMMIT


def stages(x, residual, weight, eps):
    values = {}
    values['residual_add'] = x if residual is None else x+residual
    values['float_conversion'] = values['residual_add'].float()
    values['square'] = values['float_conversion'].pow(2)
    values['mean'] = values['square'].mean(-1, keepdim=True)
    values['add_eps'] = values['mean']+eps
    values['rsqrt'] = torch.rsqrt(values['add_eps'])
    values['normalize'] = values['float_conversion']*values['rsqrt']
    values['scale'] = values['normalize']*(1.0+weight.float())
    values['cast'] = values['scale'].to(x.dtype)
    return values


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operands', type=Path)
    parser.add_argument('--json-out', required=True, type=Path)
    args = parser.parse_args()
    saved = torch.load(args.operands, weights_only=True)
    x = saved['x'].cuda()
    residual = saved['residual'].cuda() if saved['residual'] is not None else None
    weight = saved['weight'].cuda()
    packed = stages(x, residual, weight, saved['eps'])
    rowwise = [stages(x[i:i+1], residual[i:i+1] if residual is not None else None,
                      weight, saved['eps']) for i in range(x.shape[0])]
    split = {key: torch.cat([row[key] for row in rowwise]) for key in packed}
    # Replay must reproduce the actual captured norm, not just a similar formula.
    assert torch.equal(packed['cast'].cpu(), saved['packed'])
    assert torch.equal(split['cast'].cpu(), saved['rowwise'])
    record = dict(operands=str(args.operands), captured_outputs_reproduced=True,
                  shape=list(x.shape), stages=[], isolated=[], changed_elements=[])
    operations = dict(mean=lambda a: a.mean(-1, keepdim=True),
                      rsqrt=torch.rsqrt, square=lambda a: a.pow(2))
    sources = dict(mean=packed['square'], rsqrt=packed['add_eps'], square=packed['float_conversion'])
    for key, value in packed.items():
        record['stages'].append(dict(stage=key, **difference(value, split[key])))
    # Reapply each suspect unary operator to identical packed operands.
    for key, op in operations.items():
        source = sources[key]
        a = op(source)
        b = torch.cat([op(row[None]) for row in source])
        stats = difference(a, b)
        record['isolated'].append(dict(operation=key, **stats))
    for row, col in (packed['cast'] != split['cast']).nonzero().tolist():
        record['changed_elements'].append(dict(query_row=row, feature=col,
            packed=float(packed['cast'][row,col]), rowwise=float(split['cast'][row,col]),
            packed_fp32=float(packed['scale'][row,col]), rowwise_fp32=float(split['scale'][row,col]),
            mean_packed=float(packed['mean'][row,0]), mean_rowwise=float(split['mean'][row,0]),
            rsqrt_packed=float(packed['rsqrt'][row,0]), rsqrt_rowwise=float(split['rsqrt'][row,0])))
    root = Path('/home/lang/workspace/vllm')
    source = root/'vllm/ir/ops/layernorm.py'
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Unexpected pinned vLLM revision')
    subprocess.run(['git', '-C', str(root), 'diff', '--exit-code', 'HEAD', '--', str(source)], check=True)
    def register_op(fn=None, **kwargs):
        return fn if fn is not None else lambda fn: fn
    namespace = dict(torch=torch, Tensor=torch.Tensor, register_op=register_op)
    load_definitions(source, namespace, names=['rms_norm', 'fused_add_rms_norm'])
    factor = weight.float()+1.0
    def reference(a, b):
        if b is None:
            return namespace['rms_norm'](a, factor, saved['eps'])
        return namespace['fused_add_rms_norm'](a, b, factor, saved['eps'])[0]
    ref_packed = reference(x, residual)
    ref_rows = torch.cat([reference(x[i:i+1], residual[i:i+1] if residual is not None else None)
                          for i in range(x.shape[0])])
    # Explicitly compare the pre-normalization residual addition semantics.
    ref_sum = x.float() if residual is None else x.float()+residual.float()
    record['vllm_native_formula'] = dict(revision=PINNED_COMMIT,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), shared_torch_operators=True,
        tests_cuda_dispatch=False, packed_vs_local=difference(packed['cast'], ref_packed),
        rowwise_vs_local=difference(split['cast'], ref_rows),
        packed_vs_rowwise=difference(ref_packed, ref_rows),
        residual_fp32_vs_local_bf16_add=difference(packed['float_conversion'], ref_sum))
    installed_path = Path(importlib.util.find_spec('vllm').origin).parent/'model_executor/layers/layernorm.py'
    tree = ast.parse(installed_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GemmaRMSNorm')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_forward_static_with_residual')
    namespace = dict(torch=torch)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(installed_path), 'exec'), namespace)
    installed = namespace[method.name].__func__
    if residual is not None:
        eager = installed(weight, saved['eps'], x, residual)[0]
        eager_rows = torch.cat([installed(weight, saved['eps'], x[i:i+1], residual[i:i+1])[0]
                                for i in range(x.shape[0])])
        compiled = torch.compile(installed)
        fused = compiled(weight, saved['eps'], x, residual)[0]
        fused_rows = torch.cat([compiled(weight, saved['eps'], x[i:i+1], residual[i:i+1])[0]
                                for i in range(x.shape[0])])
        precision_compiled = torch.compile(installed, options={'emulate_precision_casts': True})
        precise = precision_compiled(weight, saved['eps'], x, residual)[0]
        precise_rows = torch.cat([precision_compiled(weight, saved['eps'], x[i:i+1], residual[i:i+1])[0]
                                  for i in range(x.shape[0])])
        record['installed_vllm_static_formula'] = dict(version=importlib.metadata.version('vllm'),
            source_sha256=hashlib.sha256(installed_path.read_bytes()).hexdigest(),
            shared_torch_operators=True, tests_full_engine=False,
            eager_vs_local=difference(packed['cast'], eager),
            eager_rows_vs_local=difference(split['cast'], eager_rows),
            compiled_vs_eager=difference(eager, fused),
            compiled_rows_vs_eager=difference(eager_rows, fused_rows),
            compiled_packed_vs_rows=difference(fused, fused_rows),
            compiled_vs_pinned_fp32_add=difference(ref_packed, fused),
            precision_casts_vs_eager=difference(eager, precise),
            precision_casts_rows_vs_eager=difference(eager_rows, precise_rows),
            precision_casts_packed_vs_rows=difference(precise, precise_rows))
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
