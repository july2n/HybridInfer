# 文档索引

从 [项目 README](../README.md) 了解功能与快速使用；复现实验从 [benchmark 指南](../benchmarks/README.md) 开始。以下文档按用途排列。带日期的数值是对应环境、模型和输入的历史测量，不是当前所有配置的性能承诺。`logs/` 中的原始 JSON 默认由 Git 忽略；需要核对数据时按文档中的命令重新生成。

## 当前实现

| 主题 | 文档 |
|---|---|
| 执行器、请求状态与异步调度 | [Model Runner V2](model_runner_v2.md) |
| KV 与 GDN 前缀所有权 | [KV cache](kv_cache_manager.md)、[前缀缓存](prefix_caching.md) |
| 投机公共链路与 MTP | [MTP 实现](speculative_mtp_implementation.md)、[当前进度](speculative_decoding_progress.md) |
| EAGLE-3、DFlash、DSpark | [EAGLE-3](eagle3_implementation.md)、[块草稿与真实权重实测](block_draft_implementation.md) |
| CUDA Graph 与编译 | [图与缓冲区](cuda_graph_optimization.md)、[分段编译](piecewise_compilation.md) |
| GDN 算子 | [打包卷积与状态池](gdn_kernel_optimization.md) |

## 验收与性能记录

| 主题 | 文档 |
|---|---|
| Qwen3.5 模型与 target 数值 | [模型验收](qwen35_acceptance.md)、[数值对齐](target_numerics_alignment.md) |
| MTP 接受率、质量与耗时 | [MTP-1 评估](mtp1_evaluation.md)、[概率 MTP 评估](mtp_random_evaluation.md) |
| 参考实现差分 | [vLLM 参考与适配契约](vllm_speculative_alignment.md) |
| 完整图与分段图实验 | [局部编译实验](torch_compile_experiment.md)、[完整 forward 实验](torch_compile_full_forward_experiment.md) |

## 计划与历史背景

[投机解码实施计划](speculative_decoding_plan.md) 记录设计约束和后续任务。计划中的未完成项以 [当前进度](speculative_decoding_progress.md) 为准；阅读早期性能表时，以各文档写明的日期、命令、硬件和负载为准。
