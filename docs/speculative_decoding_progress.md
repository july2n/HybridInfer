# 投机解码实施记录

## 当前可用范围

P0 基础契约和 P1 的 n-gram 贪心正确性闭环已实现，默认使用正常多词元验证，保留逐词元参考和严格对照调试模式。投机功能默认关闭；开启方式：

```python
from hybridinfer.engine.llm_engine import LLMEngine
from hybridinfer.spec_decode import SpeculativeConfig
from hybridinfer.sampling_params import SamplingParams

engine = LLMEngine(
    'models/Qwen3.5-0.8B',
    speculative=SpeculativeConfig(enabled=True, max_draft_tokens=4,
                                  ngram_min=2, ngram_max=8),
)
try:
    outputs = engine.generate(['your prompt'], SamplingParams(temperature=0))
finally:
    engine.exit()
```

仅对队列已排空、没有等待预填充且只有一个就绪解码请求的场景启用。temperature 非零、无匹配、预算不足、KV 容量不足或共享可写尾页时使用普通解码；多请求场景保留普通批处理。配置拒绝 TP>1、未知草稿后端和尚未实现的验证模式。

验证独立输入 `[anchor, d1, ..., dK]`，正常模式使用单次 K+1 行 eager 因果前向；逐词元参考模式使用普通 decode 内核。候选不进入正式 CPU/GPU token 历史。GDN 使用额外私有 slot，独立于 prefix 快照池。全接受且无截断时复制最终状态；拒绝或 EOS 截断时从未改变的正式状态通过同类因果路径重放有效输入。正式已计算长度推进到 `C + 输出数`，最后输出词元保持未计算。

KV 尾页按试算端点预留，接受后回收无效尾页，只发布已确认且已计算的完整块。请求在同步事务结束前保持 in_flight。异常时恢复正式 GDN、GPU token/长度，并使调度请求重新就绪。验证模式显式绕过普通单词元 decode 图。

## 原参考路径验证（2026-09-27）

- 基础契约、全部接受长度、EOS/长度端点、最长/最近 n-gram、重叠匹配及预算裁剪。
- 状态复制与部分接受重放、试算/重放异常恢复、非活动 slot 不变、跨页预留/回收、共享尾页及资源不足回退。
- 调度 in_flight、提交后历史与端点、拒绝历史不发布、验证图路由。
- 本地 Qwen3.5-0.8B：36 个真实 GPU 场景；84 个已提交端点的 GPU token、有效 KV、conv/recurrent state 与普通逐词元基准 **逐元素完全相等**。包括强制接受 0–4 个候选及真实 n-gram，输出长度/EOS/上下文结束，prefix 开关；开启时实际命中 prefix 快照 18 次。509-token prompt 覆盖 512 位置页边界。EOS 场景通过验证专用 logits 注入确定性 EOS，同时修改基准与投机路径。
- 硬件 RTX 3060 Ti；BF16 模型、checkpoint 原有 recurrent 精度；PyTorch 2.10.0+cu128。

复现命令（本仓库 `.runtime-deps` 为 Python 3.10 依赖）：

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  -m unittest discover -s tests -v
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_decode.py --model models/Qwen3.5-0.8B \
  --verification-mode sequential --json-out logs/validate/spec_sequential.json
```

JSON 和日志位于 `logs/validate/spec_sequential.{json,log}`。这些带 CPU 状态快照的耗时是诊断耗时，不能当作吞吐基准。`engine.model_runner.spec_metrics` 记录候选、接受、输出、试算与重放数量，以及 CUDA event 测量的状态复制、验证、恢复和 GPU 提交耗时（CPU 测试使用墙钟）。正常路径只在事务完成时等待事件，保护随后 slot/页释放；`engine.scheduler.spec_fallbacks` 记录回退原因。

## 正常模式与调试模式

正常配置 `verification_mode="packed"`（开启后的默认值）仅使用主模型批量验证的 argmax 接受/拒绝候选，不进行逐词元参考检查、不克隆 KV 用于对照、不因普通路径的浮点数值差异回退。全接受复制试算最终 GDN 状态；部分拒绝或截断重放有效输入。仅资源不足时恢复私有状态并回退到逐词元路径。

`verification_mode="sequential"` 提供正确性参考。调试配置 `verification_mode="packed_guarded"`：单次 eager 因果前向输入 K+1 个词元，投影所有预测行，随后使用相同入口状态运行逐词元参考检查。逐词元试算覆盖 native 试算 KV 尾部，只有精确一致时使用 native 预测；存在 argmax、GDN 或 KV 差异时使用参考结果和状态。显存不足也恢复私有状态后使用逐词元路径。两种路径都不发布候选历史。

真实模型再运行 36 个场景（共 84 轮验证）：最终词元及已提交状态全部与普通解码精确一致；native argmax 未发生差异，但 **84/84 轮的 GDN 和 KV 均存在数值差异，全部回退**。记录位于 `logs/validate/spec_packed_guarded.{json,log}`。这些旧结果说明逐元素完全相同不适合作为正常执行的运行时回退条件，不能把状态数值差异等同于候选拒绝。严格检查保留在调试模式。

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_decode.py --model models/Qwen3.5-0.8B \
  --verification-mode packed_guarded \
  --json-out logs/validate/spec_packed_guarded.json
```

`spec_metrics` 额外记录 packed 验证次数/输入数、词元/GDN/KV 数值差异次数、显存不足回退次数，以及 packed 和 reference 的分别耗时。`reference_trial_tokens` 明确记录额外逐词元检查的输入数，正常 packed 成功时为 0。严格调试模式的额外参考前向不进入正常路径。

## 正常 packed 验收与性能测试

`benchmarks/validate_spec_decode.py --verification-mode packed` 验证最终词元、CPU/GPU 端点，并独立从入口状态按实际提交长度运行同分块主模型参考，要求有效 KV 和 GDN 恢复结果精确一致。与普通逐词元路径的数值差异记录为 max_abs/RMSE 和有限值诊断，不作为运行时回退条件。36 个场景全部通过；全部轮次均没有数值差异回退。

`benchmarks/bench_spec_decode.py` 使用真实 n-gram、没有预知输出的草稿、没有状态快照，测量单请求离线纯解码及整段生成。默认对照启用 decode 图的普通路径，packed 验证使用 eager 路径；`--enforce-eager` 可改为两者均 eager。默认 509 个 prompt token、128 个输出 token、K=4，每种负载/模式预热 1 次、测量 3 次，记录真实接受率、回退、重放、峰值显存、基准词元匹配与耗时。结果写入 JSON，不作为服务 TTFT/P95 指标。

本次 RTX 3060 Ti / Qwen3.5-0.8B 实测中位数（普通基准启用 decode 图）：

| 固定样例 | 普通解码 tok/s | packed tok/s | 纯解码速度比 | 草稿接受率 |
|---|---:|---:|---:|---:|
| 重复文本 | 150.97 | 210.88 | 1.40× | 100% |
| Python 代码 | 156.27 | 141.35 | 0.90× | 78.05% |
| 对话提示（重复填充） | 155.02 | 215.13 | 1.39× | 100% |

三个提示均通过重复并裁剪固定为 509 token，属于可复现的小样例，不代表自然流量总体表现。每个样例/模式生成 128 token，预热一次、测量三次；全部输出与基准逐词元一致。代码样例因拒绝和重放没有加速。日志/JSON 位于 `logs/bench/spec_decode_native.json`；基准没有状态快照或预知输出草稿。

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/bench_spec_decode.py --json-out logs/bench/spec_decode_native.json
```

本次完整 GPU 回归：91 项全部通过。正常路径的状态精确比较仅在独立同分块参考验收中执行；调试模式仍保留跨路径逐元素比较。最终记录了真实接受率与性能收益，没有把旧版本的严格回退次数作为接受率。

## 后续工作

继续扩大自然文本、代码及 near-tie 验证，监测长序列数值漂移。当前正常 packed 路径不承诺对所有输入与普通解码逐词元完全一致；浮点差异可能改变接近并列的 argmax。恢复位置、已确认历史、页引用与结束条件仍严格检查。

P2 的变长批处理、GPU 批量提交、异步多词元输出，以及 P3–P7 的性能优化、随机拒绝采样和真实模型草稿均尚未实现。MTP/EAGLE/DFlash 没有创建占位模型支持。
