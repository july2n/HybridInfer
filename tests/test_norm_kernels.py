"""Norm row invariance, residual precision, input ownership and graph replay."""
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NormKernelTests(unittest.TestCase):
    @torch.inference_mode()
    def test_rows_and_permutations_preserve_outputs(self):
        from hybridinfer.layers.norm_kernels import gemma_norm
        for width in (256, 1024):
            torch.manual_seed(width)
            x = torch.randn(17, width*2, device='cuda', dtype=torch.bfloat16)[:, :width]
            residual = torch.randn_like(x)
            weight = torch.randn(width, device='cuda', dtype=torch.bfloat16)
            before = x.clone(), residual.clone()
            for prior in (None, residual):
                packed = gemma_norm(x, weight, 1e-6, prior)
                packed = packed if isinstance(packed, tuple) else (packed,)
                split = [gemma_norm(x[i:i+1], weight, 1e-6,
                         None if prior is None else prior[i:i+1]) for i in range(17)]
                split = [y if isinstance(y, tuple) else (y,) for y in split]
                permutation = torch.randperm(17, device='cuda')
                reordered = gemma_norm(x[permutation], weight, 1e-6,
                                      None if prior is None else prior[permutation])
                reordered = reordered if isinstance(reordered, tuple) else (reordered,)
                for j, output in enumerate(packed):
                    self.assertTrue(torch.equal(output, torch.cat([y[j] for y in split])))
                    self.assertTrue(torch.equal(output[permutation], reordered[j]))
                if prior is not None:
                    self.assertTrue(torch.equal(packed[1], (x.float()+prior.float()).to(x.dtype)))
            self.assertTrue(torch.equal(x, before[0]))
            self.assertTrue(torch.equal(residual, before[1]))

    @torch.inference_mode()
    def test_graph_replay_uses_new_values(self):
        from hybridinfer.layers.norm_kernels import gemma_norm
        x = torch.randn(4, 1024, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.zeros(1024, device='cuda', dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            gemma_norm(x, weight, 1e-6, residual)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = gemma_norm(x, weight, 1e-6, residual)
        x.add_(1)
        graph.replay()
        expected = gemma_norm(x, weight, 1e-6, residual)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs, expected)))


if __name__ == '__main__':
    unittest.main()
