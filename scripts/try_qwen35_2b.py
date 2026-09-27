"""Run a small text-generation smoke test with the local hybridinfer engine."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3.5-0.8B")
    parser.add_argument("--prompt", default="请用中文简单介绍一下你自己。")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = parser.parse_args()
    if not 0 < args.max_tokens < 1024:
        parser.error("--max-tokens must be between 1 and 1023")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be between 0 and 1")
    model = Path(args.model).expanduser().resolve()
    if not (model / "config.json").is_file():
        parser.error(f"Download the model first; missing {model / 'config.json'}")

    import torch
    from transformers import AutoTokenizer
    from hybridinfer.engine.llm_engine import LLMEngine
    from hybridinfer.sampling_params import SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("hybridinfer requires a working CUDA GPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if len(prompt) + args.max_tokens > 1024:
        parser.error("Prompt plus output exceeds this smoke test's 1024-token context")
    engine = LLMEngine(
        str(model),
        max_num_seqs=1,
        max_num_batched_tokens=1024,
        max_model_len=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        use_prefill_cudagraph=False,
        enable_prefix_cache=False,
    )
    try:
        result = engine.generate(
            [prompt], SamplingParams(temperature=0, max_tokens=args.max_tokens),
            use_tqdm=False,
        )[0]
        if not result["token_ids"]:
            raise RuntimeError("Generation returned no tokens")
        print(f"\nPrompt: {args.prompt}\nOutput: {result['text']}")
        print(f"Generated tokens: {len(result['token_ids'])}")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
