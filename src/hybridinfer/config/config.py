import os
from dataclasses import dataclass
from transformers import AutoConfig
from hybridinfer.spec_decode import SpeculativeConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    enable_piecewise_compile: bool = False
    enable_prefix_cache: bool = False
    prefix_cache_num_snapshots: int = 8
    hf_config: AutoConfig | None = None
    full_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    is_hybrid: bool = False
    max_state_slots: int = 0
    speculative: SpeculativeConfig | None = None

    def __post_init__(self):
        if self.enable_piecewise_compile and self.enforce_eager:
            raise ValueError("enable_piecewise_compile requires CUDA Graph execution (enforce_eager=False)")
        if self.speculative is not None and not isinstance(self.speculative, SpeculativeConfig):
            raise TypeError("speculative must be a SpeculativeConfig")
        if self.speculative and self.speculative.enabled and self.tensor_parallel_size != 1:
            raise ValueError("Speculative decoding currently requires a single GPU")
        if self.speculative and self.speculative.enabled and self.speculative.method == "mtp":
            if not self.enforce_eager or self.enable_prefix_cache:
                raise ValueError("MTP currently requires enforce_eager=True and enable_prefix_cache=False")
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8

        self.full_config = AutoConfig.from_pretrained(self.model)
        self.hf_config = getattr(self.full_config, "text_config", self.full_config)
        if getattr(self.hf_config, "model_type", None) != "qwen3_5_text":
            raise ValueError("Only Qwen3.5 Dense text models are supported")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)

        layer_types = getattr(self.hf_config, "layer_types", None)
        if layer_types is None:
            interval = getattr(self.hf_config, "full_attention_interval", 4)
            if not isinstance(interval, int) or interval < 1:
                raise ValueError("full_attention_interval must be a positive integer")
            layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                           for i in range(self.hf_config.num_hidden_layers)]
            self.hf_config.layer_types = layer_types
        if len(layer_types) != self.hf_config.num_hidden_layers or any(
            t not in ("full_attention", "linear_attention") for t in layer_types
        ):
            raise ValueError("layer_types must specify a supported type for every layer")
        self.is_hybrid = "linear_attention" in layer_types
        rope = getattr(self.hf_config, "rope_parameters", None)
        if not isinstance(rope, dict):
            rope = getattr(self.hf_config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise ValueError("Only default Qwen3.5 RoPE is supported")
        if self.enable_prefix_cache and self.prefix_cache_num_snapshots <= 0:
            raise ValueError("prefix_cache_num_snapshots must be positive")
        assert self.max_num_seqs > 0 and self.max_num_batched_tokens > 0 and self.max_model_len > 0
