"""Compare real draft backbone weights with a locally supplied z-lab reference.

The reference is supplied explicitly; no downloaded Python code is auto-executed
by the inference engine or checkpoint loader. This checks synthetic features,
not target-model generation quality.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch
from transformers import Qwen3Config
from hybridinfer.models.block_draft import DFlashDraft, DSparkDraft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['dflash', 'dspark'], required=True)
    parser.add_argument('--draft-model', required=True)
    parser.add_argument('--reference-source', required=True)
    parser.add_argument('--json-out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('dflash_reference', args.reference_source)
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    spec.loader.exec_module(reference_module)
    raw = json.loads((Path(args.draft_model)/'config.json').read_text())
    base = dict(raw.get('transformer_layer_config', raw))
    selected = (raw['dflash_config']['target_layer_ids'] if args.method == 'dflash'
                else raw['aux_hidden_state_layer_ids'])
    torch.set_default_dtype(torch.bfloat16)
    model_type = DFlashDraft if args.method == 'dflash' else DSparkDraft
    model = model_type(raw, base['hidden_size'], base['vocab_size'], len(selected))
    shared = torch.nn.Embedding(base['vocab_size'], base['hidden_size'], device='meta')
    report = model.load_checkpoint(args.draft_model, shared, shared)
    base['num_target_layers'] = raw.get('num_target_layers', 24)
    base['target_layer_ids'] = list(selected)
    if args.method == 'dspark':
        base['is_causal'] = not raw.get('sliding_window_non_causal', False)
    reference = reference_module.DFlashDraftModel(Qwen3Config(**base)).eval()
    state = model.state_dict()
    reference.load_state_dict({k:state[k] for k in reference.state_dict()}, strict=True)
    cases = []
    torch.manual_seed(42)
    with torch.inference_mode():
        for context_length, query_length in ((3, 4), (9, 2)):
            features = [torch.randn(context_length, base['hidden_size']) for _ in selected]
            noise = torch.randn(query_length, base['hidden_size'])
            positions = torch.arange(context_length, context_length+query_length)
            context = model.combine_features(features)
            hidden = noise
            for layer in model.layers:
                kv = layer.self_attn.context_kv(context, torch.arange(context_length))
                hidden = layer(hidden, positions, kv)
            actual = model.norm(hidden)
            expected = reference(position_ids=torch.arange(context_length+query_length)[None],
                noise_embedding=noise[None], target_hidden=torch.cat(features, -1)[None])[0]
            torch.testing.assert_close(actual, expected, atol=.03, rtol=.03)
            cases.append(dict(context_length=context_length, query_length=query_length,
                              max_abs=float((actual.float()-expected.float()).abs().max())))
    record = dict(method=args.method, passed=True, scope='real_checkpoint_synthetic_feature_backbone',
                  reference_source=str(Path(args.reference_source).resolve()), weight_report=report, cases=cases)
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
