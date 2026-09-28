# Target 当前数值实现与评估

更新：2026-09-29。跨浮点路径的逐位或 greedy 全序列相同仅作诊断；
数学语义、原 trial 端点与声明的局部 kernel 契约仍严格检查。

## 当前实现

- CUDA GemmaRMSNorm 使用每行固定 reduction layout 的 Triton kernel；residual
  在 FP32 相加，以未舍入值归一化，另存输入 dtype residual。CPU 保留 eager。
- 普通 decode 与 native verification convolution 使用一致的 product/累加/SiLU 顺序。
- 普通 pool GDN decode 与 packed verification 共用 recurrent scan，BV=32、FMA
  开启并写入端点。普通 decode 的 endpoint scratch 是当前开销之一。
- Conv 端点使用共享扩展历史；recurrent 保留各输入端点。

Norm 局部 oracle 为安装版 vLLM 0.19.0 static 函数，固定
`torch.compile(dynamic=False, emulate_precision_casts=False)`。
GDN/conv oracle 为固定源码提交 `a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`。
局部检查与完整引擎对照范围分开，详见 [vLLM 参考](vllm_speculative_alignment.md)。

## 当前证据与限制

MTP-1 五类输入的同输入 rolling 640 query：平均 TV 0.008659、平均 KL
0.00061625 nats，argmax 翻转 5 次，最大单位置 TV 0.050168。
详见 [MTP 评估](mtp1_evaluation.md)。最新源码清理后 repository_code rolling
128 query：平均 TV 0.010773、最大 TV 0.045765、翻转 1 次；在预先声明的
平均 TV≤0.02、最大 TV≤0.1、翻转率≤0.02 预算内。
产物为 `logs/validate/src_cleanup_numerics.json`，该预算只适用于本次配置。

概率比较使用 temperature=1 完整 softmax。诊断两边 LM head 均逐行投影，
不覆盖生产 packed LM head 的行数差异；该部分由自由生成/质量评估补充。
逐行 GEMM/attention 可用于定位执行差异，不作为性能方案。
12 道算术题普通/MTP 均正确 6 道仅是 smoke test，不能证明通用质量无损。

## 后续检查

性能优化伴随共同输入 reset/rolling 概率、层/状态偏差检查；扩大 K=2/4、B4
与生产 packed head 的自由生成和客观质量覆盖。运行前声明适用范围与预算，
未声明预算时仅报告诊断。超预算定位并修复或回退，不临时放宽门槛。
入口与命令见 [验收指南](../benchmarks/README.md)。
