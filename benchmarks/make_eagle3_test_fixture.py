"""Create UNTRAINED tiny draft weights for protocol tests, never quality/perf."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from hybridinfer.models.eagle3 import Eagle3Draft


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/Qwen3.5-0.8B')
    parser.add_argument('--output-dir', default='/tmp/hybridinfer_eagle3_UNTRAINED_fixture')
    args = parser.parse_args()
    target = AutoConfig.from_pretrained(args.model)
    target = getattr(target, 'text_config', target)
    layers = [2, target.num_hidden_layers//2, target.num_hidden_layers-3]
    cfg = dict(architectures=['LlamaForCausalLMEagle3'], model_type='llama', hidden_size=64,
               target_hidden_size=target.hidden_size, num_attention_heads=4, num_key_value_heads=2,
               num_hidden_layers=1, intermediate_size=128, rms_norm_eps=1e-6,
               max_position_embeddings=2048, vocab_size=target.vocab_size, draft_vocab_size=32,
               rope_parameters=dict(rope_type='default', rope_theta=10000000., partial_rotary_factor=.5),
               eagle_config=dict(use_aux_hidden_state=True, eagle_aux_hidden_state_layer_ids=layers),
               fixture_untrained=True)
    torch.manual_seed(42)
    model = Eagle3Draft(cfg, target.hidden_size, target.vocab_size, 3).to(dtype=torch.bfloat16)
    model.d2t.fill_(1000)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    save_file(model.state_dict(), str(directory/'model.safetensors'))
    print(f'UNTRAINED protocol fixture only: {directory}')


if __name__ == '__main__':
    main()
