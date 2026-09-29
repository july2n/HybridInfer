"""Model-native graph partitioning and compilation, independent of CUDA capture."""
from __future__ import annotations

from types import MethodType

import torch

from hybridinfer.utils.context import get_context


def _linear_attention_pre(layer, hidden, residual):
    return layer.forward_piecewise_pre(hidden, residual)


def _full_attention_pre(layer, hidden, residual):
    return layer.forward_piecewise_pre(hidden, residual)


def _other_pre(layer, hidden, residual):
    return layer.forward_piecewise_pre(hidden, residual)


def _post(layer, hidden, residual):
    return layer.forward_output(hidden, residual)


class PiecewiseBackend:
    """Compile static pieces; attention/GDN remain explicit eager boundaries."""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.callables = {}

    def compile(self, layer, kind):
        key = (id(layer), kind)
        if key not in self.callables:
            fn = layer.forward_piecewise_pre if kind == "pre" else layer.forward_output
            if self.enabled:
                # Separate entry code for different projection layouts so GDN
                # and full-attention guards don't consume one recompile budget.
                entry = _post if kind == "post" else {
                    "linear_attention": _linear_attention_pre,
                    "full_attention": _full_attention_pre,
                }.get(getattr(layer, "block_type", None), _other_pre)
                fn = torch.compile(MethodType(entry, layer), backend="inductor", fullgraph=True, dynamic=True,
                                   options={"triton.cudagraphs": False})
            self.callables[key] = fn
        return self.callables[key]

    def clear(self):
        self.callables.clear()


class SplitGraph:
    """Explicit model partitions, shared by FULL and PIECEWISE execution.

    The existing model defines the split boundaries rather than an FX rewrite.
    This keeps stateful attention/GDN kernels outside Inductor's fusion regions.
    """

    def __init__(self, manager):
        self.manager = manager
        self.backend = PiecewiseBackend(
            getattr(getattr(manager.runner, "config", None), "enable_piecewise_compile", False))
        self.wrappers = {}

    def piece(self, layer, kind):
        from .cuda_graph import CUDAGraphWrapper, CUDAGraphMode
        key = (id(layer), kind)
        if key not in self.wrappers:
            self.wrappers[key] = CUDAGraphWrapper(
                self.backend.compile(layer, kind), CUDAGraphMode.PIECEWISE)
        return self.wrappers[key]

    def __call__(self, input_ids, positions):
        model = self.manager.model
        context = get_context()
        if context.cudagraph_mode == "piecewise":
            return self.manager._run_piecewise_hidden(input_ids, positions)
        # Without compilation, NONE/FULL retain the native model operators.
        if not self.backend.enabled:
            return model(input_ids, positions)
        backbone = model.model
        hidden = backbone.embed_tokens(input_ids)
        residual = None
        for index, layer in enumerate(backbone.layers):
            if context.target_features is not None and index in context.feature_layers:
                context.target_features[index] = hidden if residual is None else hidden + residual
            projections, residual = self.piece(layer, "pre")(hidden, residual)
            hidden = layer.forward_attention_core(projections, positions, context.prefill_slices)
            hidden, residual = self.piece(layer, "post")(hidden, residual)
        hidden, _ = backbone.norm(hidden, residual)
        if context.target_features is not None:
            context.target_features["final"] = hidden
        return hidden

    def clear(self):
        for wrapper in self.wrappers.values():
            wrapper.clear()
        self.wrappers.clear()
        self.backend.clear()
