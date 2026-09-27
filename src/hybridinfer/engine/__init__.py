"""Engine components are imported on demand, without loading model kernels."""
from importlib import import_module


def __getattr__(name):
    if name in {"llm_engine", "model_runner", "decode_init", "async_output"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)
