# 投机解码实施记录

## 当前可用范围

P0 基础契约和 P1 的 n-gram **逐词元验证参考闭环**已实现。默认关闭；开启方式：

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

验证独立输入 `[anchor, d1, ..., dK]`，每个输入使用普通 eager decode 内核，候选不进入正式 CPU/GPU token 历史。GDN 使用额外私有 slot，独立于 prefix 快照池。全接受且无截断时复制最终状态；拒绝或 EOS 截断时从未改变的正式状态重放有效输入。正式已计算长度推进到 `C + 输出数`，最后输出词元保持未计算。

KV 尾页按试算端点预留，接受后回收无效尾页，只发布已确认且已计算的完整块。请求在同步事务结束前保持 in_flight。异常时恢复正式 GDN、GPU token/长度，并使调度请求重新就绪。验证模式显式绕过普通单词元 decode 图。

## 已执行验证（2026-09-27）

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
  --json-out logs/validate/spec_sequential.json
```

JSON 和日志位于 `logs/validate/spec_sequential.{json,log}`。这些带 CPU 状态快照的耗时是诊断耗时，不能当作吞吐基准。`engine.model_runner.spec_metrics` 记录候选、接受、输出、试算与重放数量，以及同步测量的状态复制、验证、恢复和 GPU 提交耗时；`engine.scheduler.spec_fallbacks` 记录回退原因。

## 尚未完成

P1 的单次多词元主模型验证仍待实现。当前每轮试算 K+1 次前向，部分接受还需要重放，不能据此宣称加速。下一步接入 eager 多词元因果验证，同时对照普通逐词元路径与同分块主模型参考；若出现 BF16 差异导致词元不一致，应保守回退或修复内核，不放宽验收容差。

P2 的变长批处理、GPU 批量提交、异步多词元输出，以及 P3–P7 的性能优化、随机拒绝采样和真实模型草稿均尚未实现。MTP/EAGLE/DFlash 没有创建占位模型支持。
