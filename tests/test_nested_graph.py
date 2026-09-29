"""Nested wrapper mode isolation and batch dispatch invariants (no GPU needed)."""
from types import SimpleNamespace
import unittest

from hybridinfer.engine.cuda_graph import (
    CUDAGraphDispatcher, CUDAGraphMode, CUDAGraphWrapper, graph_mode,
)
from hybridinfer.utils.context import BatchDescriptor, get_context, reset_context, set_context


class NestedWrapperTests(unittest.TestCase):
    def tearDown(self):
        reset_context()

    def test_full_piecewise_none_and_bucket_miss(self):
        calls = []
        inner = CUDAGraphWrapper(lambda x: calls.append('native') or x * 2,
                                 CUDAGraphMode.PIECEWISE)
        outer = CUDAGraphWrapper(lambda x: inner(x, graph_key=2), CUDAGraphMode.FULL)
        inner.graphs[2] = SimpleNamespace(replay=lambda: calls.append('piecewise'))
        inner.outputs[2] = 20
        outer.graphs[2] = SimpleNamespace(replay=lambda: calls.append('full'))
        outer.outputs[2] = 30
        set_context(False)
        with graph_mode(CUDAGraphMode.FULL):
            self.assertEqual(outer(3, graph_key=2), 30)
            self.assertEqual(outer(3, graph_key=3), 6)  # Inner must pass through.
        with graph_mode(CUDAGraphMode.PIECEWISE):
            self.assertEqual(outer(3, graph_key=2), 20)  # Outer must pass through.
        with graph_mode(CUDAGraphMode.NONE):
            self.assertEqual(outer(3, graph_key=2), 6)
        self.assertEqual(calls, ['full', 'native', 'piecewise', 'native'])
        self.assertIsNone(get_context().cudagraph_mode)
        with self.assertRaises(RuntimeError), graph_mode(CUDAGraphMode.FULL):
            raise RuntimeError('test')
        self.assertIsNone(get_context().cudagraph_mode)

    def test_dispatch_includes_composition_and_exact_decode_capacity(self):
        manager = SimpleNamespace(
            runner=SimpleNamespace(enforce_eager=False, use_prefill_cudagraph=True),
            policy=CUDAGraphMode.FULL_AND_PIECEWISE,
            decode_graphs={1: object(), 3: object()},
            prefill_graphs={128: object()}, prefill_graph_sizes=[128])
        dispatch = CUDAGraphDispatcher(manager)
        decode = BatchDescriptor('decode', 3, 3, 1, 1)
        self.assertEqual(dispatch.dispatch(decode, False, 3), CUDAGraphMode.FULL)
        self.assertEqual(dispatch.dispatch(BatchDescriptor('decode', 2, 2, 1, 1), False, 2), CUDAGraphMode.PIECEWISE)
        self.assertEqual(dispatch.dispatch(BatchDescriptor('prefill', 1, 1, None, 1), True, 1), CUDAGraphMode.PIECEWISE)
        self.assertEqual(dispatch.dispatch(BatchDescriptor('prefill', 17, 2, None, 16), True, 17), CUDAGraphMode.PIECEWISE)
        self.assertEqual(dispatch.dispatch(BatchDescriptor('prefill', 129, 1, None, 129), True, 129), CUDAGraphMode.NONE)
        self.assertEqual(dispatch.dispatch(BatchDescriptor('spec_decode', 9, 3, 3, 3), True, 9), CUDAGraphMode.NONE)
        manager.policy = CUDAGraphMode.PIECEWISE
        self.assertEqual(dispatch.dispatch(decode, False, 3), CUDAGraphMode.PIECEWISE)
        manager.policy = CUDAGraphMode.FULL
        self.assertEqual(dispatch.dispatch(BatchDescriptor('prefill', 1, 1, None, 1), True, 1), CUDAGraphMode.NONE)
        manager.runner.enforce_eager = True
        self.assertEqual(dispatch.dispatch(decode, False, 3), CUDAGraphMode.NONE)


if __name__ == '__main__':
    unittest.main()
