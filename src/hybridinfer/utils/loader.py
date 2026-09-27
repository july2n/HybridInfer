import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """Load every text parameter; only known non-text branches may be skipped."""
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    model_params = dict(model.named_parameters())
    loaded = set()
    packed_loaded = {}
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"No safetensors checkpoint found in {path}")
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for checkpoint_name in f.keys():
                # Qwen3.5 multimodal checkpoints nest the text backbone below
                # ``model.language_model`` while hybridinfer exposes it as
                # ``model``.  The visual and MTP weights are intentionally
                # outside this text-only model and must be ignored.
                if checkpoint_name.startswith(("model.visual.", "mtp.", "model.mtp.")):
                    continue
                weight_name = checkpoint_name
                if weight_name.startswith("model.language_model."):
                    weight_name = "model." + weight_name[len("model.language_model."):]
                for k in packed_modules_mapping:
                    if k in weight_name.split("."):
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model_params.get(param_name)
                        if param is None:
                            raise ValueError(f"Unexpected text weight: {checkpoint_name}")
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(checkpoint_name), shard_id)
                        packed_loaded.setdefault(param_name, set()).add(shard_id)
                        break
                else:
                    param = model_params.get(weight_name)
                    if param is None:
                        raise ValueError(f"Unexpected text weight: {checkpoint_name}")
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(checkpoint_name))
                    loaded.add(weight_name)
    for name, shard_ids in packed_loaded.items():
        expected = {shard for target, shard in packed_modules_mapping.values() if target in name.split(".")}
        if shard_ids != expected:
            raise ValueError(f"Incomplete packed weight {name}: loaded {sorted(shard_ids)}, expected {sorted(expected)}")
        loaded.add(name)
    # Tied embedding/head parameters may be distinct Parameter objects sharing
    # the same storage. Checkpoints need only contain one of the two names.
    loaded_storage = {(model_params[name].device, model_params[name].data_ptr()) for name in loaded}
    missing = [name for name, param in model_params.items()
               if (param.device, param.data_ptr()) not in loaded_storage]
    if missing:
        raise ValueError(f"Missing text weights: {', '.join(missing)}")
    return {"loaded_parameters": len(loaded), "total_parameters": len(model_params)}
