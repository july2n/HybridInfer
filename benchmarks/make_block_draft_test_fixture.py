"""Create UNTRAINED DFlash/DSpark weights for protocol tests only."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from hybridinfer.models.block_draft import DFlashDraft, DSparkDraft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--method', choices=['dflash', 'dspark'], required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    target = AutoConfig.from_pretrained(args.model)
    target = getattr(target, 'text_config', target)
    base = dict(hidden_size=target.hidden_size, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        vocab_size=target.vocab_size, layer_types=['full_attention'],
        rope_parameters=dict(rope_type='default',rope_theta=10000000.))
    if args.method == 'dflash':
        raw = dict(base, dflash_config=dict(block_size=4,mask_token_id=248070,target_layer_ids=[1,3]))
    else:
        raw = dict(transformer_layer_config=base,aux_hidden_state_layer_ids=[2,4],block_size=4,
            mask_token_id=248077,sample_from_anchor=True,draft_vocab_size=32,markov_rank=4,
            enable_confidence_head=True,confidence_head_with_markov=True)
    raw['fixture_untrained'] = True
    torch.manual_seed(42)
    model_type = DFlashDraft if args.method == 'dflash' else DSparkDraft
    model = model_type(raw,target.hidden_size,target.vocab_size,2).to(dtype=torch.bfloat16)
    if args.method == 'dspark':
        model.confidence_head.float()
        model.d2t.fill_(1000)
    weights = {k:v for k,v in model.state_dict().items() if k != 'd2t' or args.method == 'dspark'}
    directory = Path(args.output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'config.json').write_text(json.dumps(raw,indent=2)+'\n')
    save_file(weights,str(directory/'model.safetensors'))
    print(f'UNTRAINED protocol fixture only: {directory}')


if __name__ == '__main__':
    main()
