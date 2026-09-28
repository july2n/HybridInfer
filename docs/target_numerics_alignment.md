# Target 数值对齐：生产路径第一阶段

后续用户已放宽数值验收标准，并完成共享 conv 历史和 MTP-1 评估。
以下保留当时的严格对齐结果；最新决策与性能见 [MTP-1 评估](mtp1_evaluation.md)。

日期：2026-09-28。此文记录 `25fcfae` 之后的实现；前两轮实验文档中的
ordinary 算术和干预名称属于当时的代码版本。

## 已实现

- CUDA GemmaRMSNorm 使用每行独立、固定 reduction layout 的 Triton kernel。
  residual 在 FP32 相加，以未舍入结果归一化，另存输入 dtype 的 residual。
  CPU 保留原 eager 路径。
- 普通 decode convolution 与 native verification 均使用 vLLM 投机卷积的
  product/accumulation/SiLU 顺序，取消 decode 的 BF16 SiLU 前舍入。
- 普通 pool GDN decode 直接调用 packed recurrent scan：BV=32、FMA 开启，
  并保留端点写入。单独 matching BV/FMA，甚至同一源码关闭端点写入，都会
  产生约 1e-8 的 FP32 状态差；因此必须验证 state，而不只比较 BF16 output。

普通 decode 的端点 scratch 是临时分配，大小为 B×HV×DV×DK×sizeof(state)。
Qwen3.5-0.8B 的 FP32 状态每请求每层约 1 MiB。这是当前确保数值一致的代价，
尚未做性能优化或宣称加速。

## 参考范围

Norm oracle 是安装版 vLLM 0.19.0 的原始 static 函数，Torch 2.10.0+cu128，
`torch.compile(dynamic=False, emulate_precision_casts=False)`。256/1024 宽度、
1/5/17 行、带/不带 residual 的 12 个 strided-input 测试全部精确一致，
历史保存的 norm checkpoint 也一致。

自动 dynamic specialization 在切换宽度后出现另一套数值结果，保留失败产物
`target_norm_kernel_auto_dynamic.json`。上述通过结果只保证所声明的静态编译
契约，不等于完整 vLLM engine 的全部调度都已对齐。

GDN/conv oracle 仍为固定 vLLM commit
`a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`。完整生成 oracle 仍为已有的
`vllm_model_baseline_20260928.json`；两套参考不可混称为同一个版本。

## 当前 teacher-forced 结果

`repository_code`，B=1，S=5，128 次 query，eager，prefix cache off。
LM head 两边均逐行投影，因此此实验没有覆盖生产 packed LM head 的 shape 差异。

| 干预 | reset 首次翻转 | rolling 首次翻转 | 精确 logits（reset / rolling） |
|---|---:|---:|---:|
| 当前 native | #24 | #24 | 0/128 / 0/128 |
| 逐行 GEMM | #24 | #88 | 0/128 / 0/128 |
| 逐行 decode attention | #24 | #19 | 0/128 / 0/128 |
| 同时统一 GEMM、attention | 无 | 无 | 128/128 / 128/128 |

联合控制两种模式各 26/26 块的 conv/recurrent state 与有效 KV 也精确一致。
这证明本样本剩余差异可以由 GEMM 和 attention 的执行路径共同消除；
没有证明任意输入、B=4 或自由生成已精确一致。逐行调用用于定位，仍非加速实现。

## 产物及后续门槛

- `benchmarks/validate_target_norm.py`：固定源码 hash、版本与编译选项的 norm 对照。
- `tests/test_norm_kernels.py`：逐行/重排行一致性、输入不变性、CUDA graph replay。
- `tests/test_gdn_kernels.py`：普通/packed convolution 精确对照；每步 recurrent
  output 和 FP32 state 精确对照；已有 GQA、数学公式和 graph 验证。
- `logs/validate/target_alignment_shared_paths.json`：上述 8 组实际模型轨迹。
- `logs/validate/mtp_alignment_shared_paths.json`：生产路径 B1/B4 自由生成门槛。

下一阶段需要确定共享、可批量执行的 GEMM 和 attention 数值路径，同时覆盖
LM head；先做同输入/同状态检查，再做 B1/B4 自由生成和完整 vLLM gate。
native strict gate 通过前，继续保留默认 `packed_guarded`。

本轮实际自由生成结果（5 个输入，B1/B4，共 25 个请求，每请求 128 token）：

| 模式 | 与本地普通 target 全序列一致 | 与完整 vLLM 基线全序列一致 |
|---|---:|---:|
| baseline | 25/25（自身） | 6/25 |
| native packed MTP | 17/25 | 5/25 |
| packed_guarded MTP | 25/25 | 6/25 |

1,481 次 native 接受端点检查零失败，真实 MTP 执行/生命周期检查通过；
本地及 vLLM strict token gate 仍失败。此前提交版本 native 与本地为 5/25，
普通 target 与 vLLM 为 7/25。这次改善了内部路径等价性，但没有改善完整
vLLM token gate；因此当前修改仍处于数值对齐阶段，不能宣称完整兼容或加速。

最终验证：完整单元测试 127 项通过；固定 vLLM recurrent 的 72 个布局、
1,386 个端点精确一致；convolution 的 8 个布局、140 个端点精确一致。
