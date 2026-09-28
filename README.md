# HybridInfer

HybridInfer 是一个面向混合注意力模型的轻量级大语言模型推理引擎，基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 扩展实现。当前支持 Qwen3.5 Dense 纯文本推理，重点实现全注意力与门控增量网络（Gated DeltaNet）的混合执行、历史状态管理、前缀缓存和 GPU 算子优化。

项目采用 Python、PyTorch、Triton、CUDA Graph、FlashInfer 和 FLA，提供可阅读、可验证的推理实现与性能测试脚本。

## 当前功能

- **混合注意力推理**：支持 Qwen3.5 的门控分组查询注意力和 Gated DeltaNet，为每个请求维护卷积历史及递归状态，支持预填充与逐步解码。
- **请求调度**：实现连续批处理、分块预填充和预填充/解码混合执行，支持请求抢占重算、状态释放与槽位复用。
- **异步执行**：模型执行器分离输入准备、模型计算和采样；将词元、已计算长度、页表及采样状态保存在 GPU，采用增量页表更新与异步输出队列。
- **历史与前缀缓存**：全注意力采用分页键值缓存，Gated DeltaNet 使用请求级状态池；通过块对齐的状态快照联合复用两类历史，支持并发读取与缓存淘汰。前缀缓存默认关闭，可按需启用。
- **CUDA Graph**：解码采用完整模型图，预填充按词元数量分桶捕获静态投影、归一化和前馈网络，动态注意力核心保持普通执行；复用层间缓冲区与图内存池。
- **Triton 算子**：直接处理变长请求的打包卷积，按请求槽位原地读写 FP32 递归状态，减少填充、拼接及状态搬运；普通与分段预填充共用核心实现。

## 实测性能

测试环境为 **RTX 3060 Ti、Qwen3.5-0.8B、BF16 计算、FP32 递归状态**，开启解码图和分段预填充图。以下为各测试场景多次运行后的中位耗时：

| 场景 | 耗时 |
|---|---:|
| 单请求，512 个输入词元 | 36.85 毫秒 |
| 4 个请求，各 128 个输入词元 | 36.90 毫秒 |
| 变长预填充，总计 480 个输入词元 | 37.60 毫秒 |
| 分块变长预填充，总计 960 个输入词元 | 71.90 毫秒 |
| 7 个解码请求与延后到达的预填充请求混合执行 | 90.70 毫秒 |

纯预填充每项测量 10 次，混合负载测量 3 次；混合项为整段负载耗时。这些结果不是在线服务的首词元延迟或逐词元延迟，也未与外部框架进行同条件性能对照。完整配置、复现命令和算子测量见 [GDN 内核优化记录](docs/gdn_kernel_optimization.md)。

## 安装与使用

需要可用的 NVIDIA GPU 和 CUDA 环境，以及与当前环境兼容的 PyTorch、Triton、Transformers、FlashInfer、safetensors 和 tqdm。当前项目配置尚未声明完整运行依赖，以下命令仅安装项目包；运行依赖需在环境中预先准备。

```bash
pip install -e .
```

Python 包名和命令名均为 `hybridinfer`。`hybridinfer` 命令目前只输出版本；文本生成使用 Python 接口或脚本。

通过 Python 接口可启用图执行与前缀缓存：

```python
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.sampling_params import SamplingParams

engine = LLMEngine(
    "models/Qwen3.5-0.8B",
    max_num_seqs=8,
    max_num_batched_tokens=1024,
    max_model_len=1024,
    gpu_memory_utilization=0.75,
    enforce_eager=False,
    use_prefill_cudagraph=True,
    enable_prefix_cache=True,
    prefix_cache_num_snapshots=8,
)
try:
    tokens = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": "请介绍混合注意力。"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    result = engine.generate(
        [tokens],
        SamplingParams(temperature=0, max_tokens=64),
        use_tqdm=False,
    )
    print(result[0]["text"])
finally:
    engine.exit()
```

测试脚本可通过 `HYBRIDINFER_MODEL` 指定默认模型路径。`HYBRIDINFER_GDN_DECODE_BACKEND=flashinfer` 可切换递归解码后端；该设置需在引擎初始化前生效。默认 FP32 状态与支持的维度布局使用 Triton 状态池内核。

## 实现文档

- [模型执行器与异步状态管理](docs/model_runner_v2.md)
- [混合模型前缀缓存](docs/prefix_caching.md)
- [CUDA Graph 与缓冲区管理](docs/cuda_graph_optimization.md)
- [GDN 内核与共享预填充实现](docs/gdn_kernel_optimization.md)
- [投机解码实施计划](docs/speculative_decoding_plan.md)
- [投机解码实现进度与验收边界](docs/speculative_decoding_progress.md)
- [模型数值验收记录](docs/qwen35_acceptance.md)

## 后续计划

- [x] 单请求 n-gram 贪心投机解码执行链路（默认关闭；开启后保守验证，native packed 仍为实验选项）。
- [x] 变长多请求 n-gram 贪心验证、GPU 接受/提交与异步输出句柄（GDN 重放仍需取回长度，完整系统验收见实施记录）。
- [ ] 随机投机采样及 MTP/EAGLE/DFlash 草稿。
- [ ] 支持 MoE 模型。
