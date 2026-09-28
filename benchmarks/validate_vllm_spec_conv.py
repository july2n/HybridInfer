"""Pinned vLLM speculative convolution, rolling histories and endpoint windows."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import torch
import triton
import triton.language as tl
from validate_vllm_spec_alignment import load_definitions, PINNED_COMMIT
from hybridinfer.layers.gdn_kernels import packed_causal_conv, conv_endpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-root', default='/home/lang/workspace/vllm')
    parser.add_argument('--json-out', default='logs/validate/vllm_spec_conv.json')
    args = parser.parse_args()
    root = Path(args.vllm_root)
    source = root/'vllm/model_executor/layers/mamba/ops/causal_conv1d.py'
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Unexpected vLLM revision')
    if subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--', str(source)]).returncode:
        raise ValueError('Modified vLLM convolution source')
    ns = dict(torch=torch, triton=triton, tl=tl, NULL_BLOCK_ID=0,
              current_platform=SimpleNamespace(is_arch_support_pdl=lambda: False))
    load_definitions(source, ns, ['_causal_conv1d_update_kernel', 'causal_conv1d_update'])
    reference = ns['causal_conv1d_update']
    record = dict(passed=True, revision=PINNED_COMMIT,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), layouts=[])
    for dtype in [torch.bfloat16, torch.float16]:
        for lengths in [[1], [1, 2, 4, 9], [9, 1, 3], [5]*8]:
            torch.manual_seed(42)
            total, channels, n = sum(lengths), 137, len(lengths)
            x = torch.randn(total, channels, device='cuda', dtype=dtype)
            weights = torch.randn(channels, 1, 4, device='cuda', dtype=dtype)
            pool = torch.randn(n+2, channels, 3, device='cuda', dtype=dtype)
            slots = torch.arange(n, device='cuda').flip(0)+1
            reference_pool = torch.zeros(n+2, channels, 3+max(lengths)-1, device='cuda', dtype=dtype)
            reference_pool[:, :, :3] = pool
            boundaries = [0]
            for length in lengths:
                boundaries.append(boundaries[-1]+length)
            cu = torch.tensor(boundaries, device='cuda', dtype=torch.int32)
            endpoints = conv_endpoints(x, pool, slots, cu)
            output = packed_causal_conv(x, weights, pool, slots, cu, max(lengths), round_before_silu=False)
            expected = reference(x.clone(), reference_pool, weights[:, 0], activation='silu',
                conv_state_indices=slots.to(torch.int32), num_accepted_tokens=torch.ones(n, device='cuda', dtype=torch.int32),
                query_start_loc=cu, max_query_len=max(lengths), out=torch.empty_like(x))
            histories_match = True
            for row, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
                for t in range(end-start):
                    histories_match &= torch.equal(endpoints[start+t], reference_pool[slots[row], :, t:t+3])
            outputs_match = torch.equal(output, expected)
            record['passed'] &= histories_match and outputs_match
            record['layouts'].append(dict(lengths=lengths, dtype=str(dtype), endpoints=total,
                histories_match=histories_match, outputs_match=outputs_match,
                output_max_abs=(output-expected).abs().max().item()))
    Path(args.json_out).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record['passed'] else 1)


if __name__ == '__main__':
    main()
