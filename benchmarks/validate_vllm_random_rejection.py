"""Pinned vLLM random acceptance kernel versus shared rejection decisions."""
import argparse
import json
from pathlib import Path
import torch
from validate_vllm_spec_alignment import load_definitions, load_reference, PINNED_COMMIT
from hybridinfer.spec_decode.rejection import rejection_decisions
import triton
import triton.language as tl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vllm-root', default='/home/lang/workspace/vllm')
    parser.add_argument('--json-out', default='logs/validate/vllm_random_rejection.json')
    args = parser.parse_args()
    root = Path(args.vllm_root)
    import subprocess
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PINNED_COMMIT:
        raise ValueError('Unexpected vLLM reference revision')
    _, _, _, hashes = load_reference(root)
    ns = dict(torch=torch, triton=triton, tl=tl)
    load_definitions(root/'vllm/v1/sample/rejection_sampler.py', ns, ['rejection_random_sample_kernel'])
    kernel = ns['rejection_random_sample_kernel']
    torch.manual_seed(42)
    batches = requests = 0
    for point_mass in [False, True]:
        for trial in range(100):
            lengths = [trial % 9, 8, 1, 3, 0]
            total, vocab = sum(lengths), 17
            candidates = torch.randint(vocab, (total,), device='cuda')
            p = torch.rand(total, vocab, device='cuda').softmax(-1)
            q = torch.rand_like(p).softmax(-1)
            draws = torch.rand(total, device='cuda')
            if not point_mass:
                # Include impossible draft values and exact acceptance ties.
                q[0, candidates[0]] = 0
                draws[1] = p[1, candidates[1]]/q[1, candidates[1]]
            recovered = torch.arange(total, device='cuda', dtype=torch.int64)+100
            bonus = torch.arange(len(lengths), device='cuda', dtype=torch.int64)+1000
            ends = torch.tensor(lengths, device='cuda', dtype=torch.int32).cumsum(0).to(torch.int32)
            output = torch.full((len(lengths), 9), -1, dtype=torch.int64, device='cuda')
            greedy = torch.zeros(len(lengths), device='cuda', dtype=torch.bool)
            kernel[(len(lengths),)](output, ends, candidates, None if point_mass else q,
                p, bonus, recovered, draws, greedy, 8, vocab, None,
                NO_DRAFT_PROBS=point_mass, SYNTHETIC_MODE=False)
            chosen_p = p.gather(1, candidates[:, None])[:, 0]
            chosen_q = torch.ones_like(chosen_p) if point_mass else q.gather(1, candidates[:, None])[:, 0]
            decisions = rejection_decisions(chosen_p, chosen_q, draws).cpu().tolist()
            candidates_cpu, recovered_cpu, bonus_cpu = candidates.cpu().tolist(), recovered.cpu().tolist(), bonus.cpu().tolist()
            expected = torch.full_like(output, -1)
            start = 0
            for row, count in enumerate(lengths):
                for step in range(count):
                    if decisions[start+step]:
                        expected[row, step] = candidates_cpu[start+step]
                    else:
                        expected[row, step] = recovered_cpu[start+step]
                        break
                else:
                    expected[row, count] = bonus_cpu[row]
                start += count
            if not torch.equal(output, expected):
                raise AssertionError(f'vLLM rejection mismatch trial={trial} point_mass={point_mass}')
            batches += 1
            requests += len(lengths)
    record = dict(passed=True, revision=PINNED_COMMIT, source_hashes=hashes,
                  batches=batches, requests=requests, point_mass_and_probabilistic_q=True,
                  note='Uniforms and recovered/bonus tokens supplied identically; RNG algorithms differ.')
    Path(args.json_out).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
