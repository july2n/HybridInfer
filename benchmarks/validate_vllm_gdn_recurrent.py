"""Original pinned vLLM multi-token GDN kernel and all acceptance endpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import torch
import triton
import triton.language as tl
from validate_vllm_spec_alignment import load_definitions, PINNED_COMMIT
from hybridinfer.layers.gdn_kernels import packed_gdn_recurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-root', default='/home/lang/workspace/vllm')
    parser.add_argument('--json-out', default='logs/validate/vllm_gdn_recurrent.json')
    args = parser.parse_args()
    root = Path(args.vllm_root)
    source = root/'vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py'
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Unexpected vLLM revision')
    if subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', str(source)]).returncode:
        raise ValueError('Modified vLLM recurrent reference')
    ns = dict(torch=torch, triton=triton, tl=tl)
    load_definitions(source, ns)
    reference = ns['fused_sigmoid_gating_delta_rule_update']
    record = dict(passed=False, numerical_passed=True, bitwise_passed=True, revision=PINNED_COMMIT,
                  tolerance=dict(state_rtol=1e-4, state_atol=1e-5, output_rtol=.02, output_atol=.002),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), layouts=[])
    for hq, hv in [(2, 4), (4, 4), (16, 16)]:
        for lengths in [[1], [1, 2, 4, 9], [9, 1, 3], [5]*8]:
            for initial_bias in range(1, max(lengths)+1):
                torch.manual_seed(42)
                n, total = len(lengths), sum(lengths)
                q = torch.randn(total, hq, 128, device='cuda', dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn(total, hv, 128, device='cuda', dtype=torch.bfloat16)
                a = torch.randn(total, hv, device='cuda', dtype=torch.bfloat16)
                b = torch.randn_like(a)
                log = torch.randn(hv, device='cuda')
                bias = torch.randn(hv, device='cuda', dtype=torch.bfloat16)
                pool = torch.randn(n+2, hv, 128, 128, device='cuda')*.1
                slots = torch.arange(n, device='cuda').flip(0)+1
                initial = pool.index_select(0, slots).clone()
                boundaries = [0]
                for count in lengths:
                    boundaries.append(boundaries[-1]+count)
                cu = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
                expected_pool = torch.zeros(total+1, hv, 128, 128, device='cuda')
                indices = torch.zeros(n, max(lengths), device='cuda', dtype=torch.int32)
                accepted = torch.tensor([min(initial_bias, count) for count in lengths], device='cuda', dtype=torch.int32)
                for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                    indices[row, :end-start] = torch.arange(start+1, end+1, device='cuda')
                    expected_pool[start+int(accepted[row])] = initial[row]
                out, snapshots = packed_gdn_recurrent(q, k, v, a, b, log, bias, pool, slots, cu)
                expected_out, expected_state = reference(log, a[None], b[None], bias, q[None], k[None], v[None],
                    initial_state=expected_pool, inplace_final_state=True, cu_seqlens=cu,
                    ssm_state_indices=indices, num_accepted_tokens=accepted, use_qk_l2norm_in_kernel=True)
                expected_snapshots = expected_state[1:]
                error = (snapshots-expected_snapshots).abs().max().item()
                numerical = torch.allclose(snapshots, expected_snapshots, rtol=1e-4, atol=1e-5)
                numerical &= torch.allclose(out, expected_out[0], rtol=.02, atol=.002)
                bitwise = torch.equal(snapshots, expected_snapshots) and torch.equal(out, expected_out[0])
                record['numerical_passed'] &= numerical
                record['bitwise_passed'] &= bitwise
                record['layouts'].append(dict(lengths=lengths, initial_bias=initial_bias, key_heads=hq, value_heads=hv,
                                              endpoints=total, state_max_abs=error,
                                              numerical_passed=numerical, bitwise_passed=bitwise))
    record['passed'] = record['numerical_passed']
    Path(args.json_out).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
