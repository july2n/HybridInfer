"""Compare production Qwen3.5 logits with an independent Transformers reference.

Reference and engine run in separate processes to fit an 8 GB GPU. Teacher
forcing uses the reference continuation so every compared row has identical
history; independent greedy generation is checked separately. The reference
uses Transformers' eager attention and PyTorch GDN implementations, rather
than sharing hybridinfer's kernels or weight loader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

from engine_bench_utils import MODES, env_info, make_engine, make_params, run_until_idle, save_json
from hybridinfer.engine.sequence import Sequence


def workloads(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = []
    for i, text in enumerate(("请用中文简单介绍一下你自己。", "What is 7 times 8?", "Write a short Python function to add two numbers.")):
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        cases.append({"name": f"chat_{i}", "prompt": tokens})
    # Ordinary text tokens avoid randomly selecting image/control placeholders.
    for length in (1, 63, 64, 65, 127, 129, 257):
        generator = torch.Generator().manual_seed(length)
        cases.append({"name": f"boundary_{length}", "prompt": torch.randint(0, 10000, (length,), generator=generator).tolist()})
    return cases


@torch.inference_mode()
def reference(args):
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf
    from transformers import AutoConfig
    from safetensors import safe_open

    config = AutoConfig.from_pretrained(args.model, local_files_only=True).text_config
    config._attn_implementation = "eager"
    # Explicit independent reference: no optional FLA/causal-conv kernel imports.
    hf.FusedRMSNormGated = None
    # Construct in BF16 like from_pretrained(dtype=...), rather than casting
    # the whole model afterward: .to(BF16) would also round RoPE's FP32
    # inverse-frequency buffers and corrupt the reference positions.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = hf.Qwen3_5TextModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    if model.rotary_emb.inv_freq.dtype != torch.float32:
        raise AssertionError("Reference RoPE frequencies must stay FP32")
    params = dict(model.named_parameters())
    loaded = set()
    head = None
    precision_weights = {}
    for file in sorted(Path(args.model).glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name == "lm_head.weight":
                    head = checkpoint.get_tensor(name)
                prefix = "model.language_model."
                if name.startswith(prefix):
                    target = name[len(prefix):]
                    if target not in params:
                        raise ValueError(f"Unexpected text weight: {name}")
                    params[target].copy_(checkpoint.get_tensor(name))
                    if args.reference_profile == "kernel" and (target.endswith(".A_log") or target.endswith("linear_attn.norm.weight")):
                        precision_weights[target] = checkpoint.get_tensor(name)
                    loaded.add(target)
    missing = set(params) - loaded
    if missing:
        raise ValueError(f"Reference missing weights: {sorted(missing)}")
    for name, weight in precision_weights.items():
        params[name].data = weight
    if args.reference_profile == "kernel":
        # Independent PyTorch implementation with FLA's normalization precision:
        # FP32 reduction, then cast normalized q/k to their original dtype.
        def fp32_l2norm(x, dim=-1, eps=1e-6):
            normalized = x.float() * torch.rsqrt(x.float().square().sum(dim=dim, keepdim=True) + eps)
            return normalized.to(x.dtype)
        hf.l2norm = fp32_l2norm
    model = model.cuda().eval()
    head = model.embed_tokens.weight if config.tie_word_embeddings else head.cuda().to(torch.bfloat16)
    for layer in model.layers:
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            gdn.causal_conv1d_fn = None
            gdn.causal_conv1d_update = hf.torch_causal_conv1d_update
            gdn.chunk_gated_delta_rule = hf.torch_chunk_gated_delta_rule
            gdn.recurrent_gated_delta_rule = hf.torch_recurrent_gated_delta_rule
    cases = workloads(args.model)
    for case in cases:
        inputs = torch.tensor([case["prompt"]], device="cuda")
        cache = None
        rows, tokens = [], []
        layer_inputs, layer_outputs, handles = [], [], []
        if case["name"] == "boundary_1":
            for layer in model.layers:
                handles.append(layer.register_forward_pre_hook(
                    lambda module, args, kwargs: layer_inputs.append((args[0] if args else kwargs["hidden_states"]).detach().cpu()),
                    with_kwargs=True))
                handles.append(layer.register_forward_hook(
                    lambda module, args, output: layer_outputs.append(output.detach().cpu())))
        for step in range(args.steps):
            output = model(inputs, past_key_values=cache, use_cache=True)
            if step == 0:
                for handle in handles:
                    handle.remove()
            cache = output.past_key_values
            logits = torch.nn.functional.linear(output.last_hidden_state[:, -1], head)
            token = int(logits.argmax(-1).item())
            rows.append(logits[0].float().cpu())
            tokens.append(token)
            inputs = torch.tensor([[token]], device="cuda")
        case.update(tokens=tokens, logits=torch.stack(rows))
        if layer_inputs:
            case.update(layer_inputs=layer_inputs, layer_outputs=layer_outputs)
        print(f"reference {case['name']} done", flush=True)
    torch.save({"meta": json.loads(json.dumps(env_info(), default=str)), "cases": cases, "steps": args.steps,
                "reference": "Transformers eager attention / torch GDN / cached decode",
                "reference_profile": args.reference_profile}, args.reference)


class CaptureSampler(nn.Module):
    def __init__(self, forced=None):
        super().__init__()
        self.forced = forced
        self.rows = []

    def forward(self, logits, temperatures):
        self.rows.append(logits[0].detach().float().cpu())
        if self.forced is None:
            return logits.argmax(-1)
        return torch.tensor([self.forced[len(self.rows) - 1]], device=logits.device)


def compare_rows(reference_rows, candidate_rows):
    if reference_rows.shape != candidate_rows.shape:
        raise ValueError("Teacher-forced logits shape mismatch")
    delta = candidate_rows - reference_rows
    finite = bool(torch.isfinite(candidate_rows).all() and torch.isfinite(reference_rows).all())
    rmse = delta.square().mean(-1).sqrt()
    relative = rmse / reference_rows.square().mean(-1).sqrt().clamp_min(1e-6)
    per_step = []
    for step, (a, b) in enumerate(zip(reference_rows, candidate_rows)):
        ta, tb = int(a.argmax()), int(b.argmax())
        # Require BOTH competing tokens to be within one BF16 ULP in both
        # distributions; an unrelated close runner-up cannot hide a mismatch.
        tolerance = max(0.01, max(abs(float(a[ta])), abs(float(b[tb]))) / 128)
        tie = ta != tb and abs(float(a[ta] - a[tb])) <= tolerance and abs(float(b[ta] - b[tb])) <= tolerance
        per_step.append({"step": step, "rmse": float(rmse[step]),
                         "relative_rmse": float(relative[step]),
                         "max_abs": float(delta[step].abs().max()),
                         "reference_token": ta, "engine_token": tb,
                         "numeric_tie": tie, "tie_tolerance": tolerance,
                         "cosine_similarity": float(torch.nn.functional.cosine_similarity(a, b, dim=0))})
    # Fixed acceptance bounds, declared independently of this run's results.
    passed = finite and bool((relative <= 0.02).all()) and float(delta.abs().max()) <= 1.0
    passed &= all(row["reference_token"] == row["engine_token"] or row["numeric_tie"] for row in per_step)
    return {"pass": passed, "finite": finite, "rmse": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()), "steps": per_step}


@torch.inference_mode()
def engine(args):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        data = torch.load(args.reference, map_location="cpu", weights_only=True)
    runner = make_engine(args.model, MODES[0], max_num_seqs=1,
                         max_num_batched_tokens=512, max_model_len=512,
                         gpu_memory_utilization=0.7)
    report = {"meta": env_info(), "reference_meta": data["meta"],
              "reference": data["reference"],
              "reference_profile": data.get("reference_profile", "native"),
              "limits": {"relative_rmse": 0.02, "max_abs": 1.0}, "cases": []}
    try:
        for case in data["cases"]:
            forced = CaptureSampler(case["tokens"])
            runner.model_runner.sampler = forced
            seq = Sequence(case["prompt"], make_params(data["steps"]))
            runner.scheduler.add(seq)
            run_until_idle(runner, [seq])
            comparison = compare_rows(case["logits"], torch.stack(forced.rows))
            free = CaptureSampler()
            runner.model_runner.sampler = free
            seq = Sequence(case["prompt"], make_params(data["steps"]))
            runner.scheduler.add(seq)
            run_until_idle(runner, [seq])
            tokens = seq.completion_token_ids
            first = next((i for i, (a, b) in enumerate(zip(case["tokens"], tokens)) if a != b), None)
            generation_accepted = len(tokens) == len(case["tokens"]) and (
                first is None or comparison["steps"][first]["numeric_tie"])
            item = {"name": case["name"], "prompt_length": len(case["prompt"]),
                    "teacher_forced": comparison, "reference_tokens": case["tokens"],
                    "engine_tokens": tokens, "generation_exact": tokens == case["tokens"],
                    "first_generation_mismatch": first, "generation_accepted": generation_accepted,
                    "pass": comparison["pass"] and generation_accepted}
            report["cases"].append(item)
            print(f"{case['name']}: pass={item['pass']} exact={item['generation_exact']} rmse={comparison['rmse']:.6f} max={comparison['max_abs']:.6f}", flush=True)
        # Independently feed each decoder the SAME reference input at position
        # zero to distinguish local operator differences from accumulated drift.
        probe_case = next(case for case in data["cases"] if "layer_inputs" in case)
        checks, handles = [], []
        for index, layer in enumerate(runner.model_runner.model.model.layers):
            expected_input = probe_case["layer_inputs"][index][0].cuda()
            expected_output = probe_case["layer_outputs"][index][0].float()
            def inject(module, inputs, value=expected_input):
                positions, hidden, residual, slices = inputs
                return positions, value, None, slices
            def capture(module, inputs, output, expected=expected_output, index=index):
                hidden, residual = output
                actual = (hidden + residual).float().cpu()
                delta = actual - expected
                checks.append({"layer": index, "type": module.block_type,
                               "rmse": float(delta.square().mean().sqrt()),
                               "max_abs": float(delta.abs().max()),
                               "relative_rmse": float(delta.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-6))})
            handles.append(layer.register_forward_pre_hook(inject))
            handles.append(layer.register_forward_hook(capture))
        try:
            runner.model_runner.sampler = CaptureSampler()
            seq = Sequence(probe_case["prompt"], make_params(1))
            runner.scheduler.add(seq)
            run_until_idle(runner, [seq])
        finally:
            for handle in handles:
                handle.remove()
        report["isolated_decoder_checks"] = checks
    finally:
        runner.exit()
    report["overall_pass"] = all(case["pass"] for case in report["cases"])
    report["overall_exact_generation"] = all(case["generation_exact"] for case in report["cases"])
    report["max_relative_rmse"] = max(row["relative_rmse"] for case in report["cases"] for row in case["teacher_forced"]["steps"])
    report["max_abs"] = max(case["teacher_forced"]["max_abs"] for case in report["cases"])
    save_json(args.json_out, report)
    if not report["overall_pass"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.5-0.8B")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--phase", choices=("parent", "reference", "engine"), default="parent")
    parser.add_argument("--reference", default="logs/validate/dense_reference.pt")
    parser.add_argument("--reference-profile", choices=("native", "kernel"), default="native",
                        help="native BF16 fallback, or FP32 checkpoint scalars / FLA-style qk normalization")
    parser.add_argument("--json-out", default="logs/validate/dense_reference.json")
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 128:
        parser.error("steps must be between 1 and 128")
    Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
    if args.phase == "reference":
        reference(args)
    elif args.phase == "engine":
        engine(args)
    else:
        for phase in ("reference", "engine"):
            child = subprocess.run([sys.executable, __file__, "--phase", phase,
                            "--model", args.model, "--steps", str(args.steps),
                            "--reference", args.reference, "--json-out", args.json_out,
                            "--reference-profile", args.reference_profile])
            if child.returncode:
                raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
