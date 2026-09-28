"""Small objective quality smoke test; complements numerical and timing tests."""
import argparse
import json
import re
from pathlib import Path

import torch
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams
from hybridinfer.spec_decode import SpeculativeConfig


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-out', default='logs/validate/mtp1_quality.json')
    args = parser.parse_args()
    spec = SpeculativeConfig(enabled=True, method='mtp', max_draft_tokens=1,
                             verification_mode='packed')
    engine = LLMEngine('models/Qwen3.5-0.8B', max_num_seqs=4, max_model_len=256,
        max_num_batched_tokens=512, gpu_memory_utilization=.6, enforce_eager=True,
        enable_prefix_cache=False, speculative=spec)
    tasks = [('17 + 28', 45), ('83 - 37', 46), ('12 * 7', 84), ('144 / 12', 12),
             ('23 + 49', 72), ('91 - 58', 33), ('8 * 9', 72), ('125 / 5', 25),
             ('46 + 37', 83), ('72 - 29', 43), ('13 * 6', 78), ('168 / 14', 12)]
    prompts = [engine.tokenizer.apply_chat_template([
        dict(role='user', content=f'Calculate {expression}. Return only the integer answer.')],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
        for expression, _ in tasks]
    prompts = [p['input_ids'] if hasattr(p, 'keys') else p for p in prompts]
    record = dict(completed=False, scope='12 fixed arithmetic tasks; quality smoke test only', cases=[])
    try:
        for mode in ('baseline', 'packed'):
            config = None if mode == 'baseline' else spec
            engine.config.speculative = engine.scheduler.speculative = config
            output = engine.generate(prompts, SamplingParams(temperature=0, max_tokens=64), use_tqdm=False)
            for (expression, answer), result in zip(tasks, output):
                text = engine.tokenizer.decode(result['token_ids'], skip_special_tokens=True)
                visible = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()
                correct = bool(re.fullmatch(r'\s*'+str(answer)+r'\s*', visible))
                record['cases'].append(dict(mode=mode, expression=expression, expected=answer,
                    text=text, correct=correct, tokens=result['token_ids']))
        record['completed'] = True
        record['correct'] = {mode: sum(r['correct'] for r in record['cases'] if r['mode'] == mode)
                             for mode in ('baseline', 'packed')}
    finally:
        engine.exit()
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record['correct']))


if __name__ == '__main__':
    main()
