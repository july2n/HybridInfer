# 投机解码实施记录

## 当前可用范围

P0 基础契约、P1 单请求链路与 P2 首版变长 n-gram 贪心批处理已接入。投机功能默认关闭；开启后默认使用保守 `packed_guarded` 验证，保留逐词元参考和显式选择的 native packed 模式。开启方式：

**2026-09-28 更新：native packed 的扩展词元一致性验收未通过。** 中文、真实仓库代码及低匹配输入已复现与普通解码的词元分歧，不能再仅凭原 36 个重复样例声明 native packed 已满足完整 P1 验收。执行链路和同分块状态恢复仍通过；开启后的默认配置改为 `packed_guarded`，以保守回退保持普通解码的词元和状态。这个模式额外计算参考路径，不承诺加速；`sequential` 可以直接执行参考路径。P2 当前交付及限制见下文新增记录。

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

仅对队列已排空、没有等待预填充的就绪解码请求启用；单请求保留原入口，多请求使用变长批次。temperature 非零、整批无匹配、预算不足、KV 容量不足或共享可写尾页时使用普通解码。批次内 K_i=0 的请求保留锚点行，可与有草稿请求共同验证。配置拒绝 TP>1、未知草稿后端和尚未实现的验证模式。

验证独立输入 `[anchor, d1, ..., dK]`，正常模式使用单次 K+1 行 eager 因果前向；逐词元参考模式使用普通 decode 内核。候选不进入正式 CPU/GPU token 历史。GDN 为每个批量验证请求使用独立私有 slot，状态池在功能开启时预留与正式 slots 等量的试算 slots，独立于 prefix 快照池。全接受且无截断时复制最终状态；拒绝或 EOS 截断时从未改变的正式状态通过同类因果路径重放有效输入。正式已计算长度推进到 `C + 输出数`，最后输出词元保持未计算。

KV 尾页按试算端点预留，接受后回收无效尾页，只发布已确认且已计算的完整块。单请求等待同步事务结束；批量请求等待独立输出句柄的完成事件后才离开 in_flight。异常时恢复正式 GDN、GPU token/长度，并使调度请求重新就绪。验证模式显式绕过普通单词元 decode 图。

## P2 首版代码交付（2026-09-28）

新增 `metadata.py` 定义完整 hidden-state 与选中 logits 两套坐标系；候选、采样行分别维护累计长度。`batch_execution.py` 对变长 `[u,d1,...,dK_i]` 运行一次因果 target 前向并投影全部有效行。`batch_verifier.py` 在 GPU 上求连续接受前缀、首个拒绝 correction、全接受 bonus、EOS/长度截断和有效输出长度；`commit.py` 只将有效词元与端点写入正式 GPU 历史。输出 `[B,max(K_i)+1]` 的占位值不会进入 Sequence。

调度器按全批 token 预算裁剪草稿，并为后续请求保留锚点预算。KV 预留失败会撤销整批已预留尾页；成功后请求整体进入 in_flight。异步句柄拥有独立 payload、pinned CPU 缓冲区和完成事件，消费后按每请求真实输出数提交 CPU 长度和释放页/slots。显式 native 部分接受只按有效输入重放一次打包前向，不再采样。该恢复仍需取回接受长度，尚未实现完全在 GPU 上构造恢复输入。

**多请求保守模式的边界与修正：** 初版把每请求逐个以 batch=1 复核，32-token 的 8 个短场景通过，但 128-token、batch=3/4/8、prefix 开/关的 12 个场景全部出现部分词元分歧；556 个同路径恢复端点仍通过。记录保留在 `logs/validate/spec_batch_guarded_final.{json,log}`，标记 `passed=false`。这是批次形状改变带来的真实失败，不能用短输出通过代替长输出验收。

修正后，多请求 `sequential` 与 `packed_guarded` 以普通解码相同的 batch 形状计算一次锚点，只提交每请求一个目标 token 和对应锚点状态。`packed_guarded` 仍试算 K_i+1 行，但不会提交后续 packed 候选状态；该回退计为零草稿接受，首个候选恰好匹配另记 `reference_first_draft_matches`。只对 K_i=0 比较相同端点的 packed/reference 状态，K_i>0 的不同端点不会误计为状态不一致。多词元 fast path 仅在显式 `packed` 中启用，不能把保守批处理宣称为加速。

新增测试覆盖 `[3,0,2]`、`[0,1,4]` 和全零候选批次、每个接受长度、拒绝后的后续行匹配、EOS/上下文/输出限制、CUDA 接受与提交、slot 重排、独立输出句柄、GPU/CPU 端点、整批异常回滚、OOM、预留失败、完成事件消费、独立预填充与后续抢占重算。真实模型检查使用 `benchmarks/validate_spec_batch.py`，逐批比较正式历史与同路径 KV/conv/recurrent 恢复状态；长 prefix 模式增加实际缓存命中和跨页，最终状态与普通批量基准另外逐元素比较。

当前 GPU 回归 **112 项全部通过，无跳过**，日志为 `/tmp/nano_qwen_p2_final_tests_gpu.log`。包含按普通 batch 形状推进锚点的回归，防止恢复为逐请求 batch=1 复核。

真实 Qwen3.5-0.8B、普通 decode 图开启、长 prompt 509 token 的保守批处理验收：batch=4/8 × prefix 开/关，共 **4 个场景、247 个批次、1467 个请求端点通过**，每请求输出约 64 token（长度错开）。输出 token、最终有效 KV/conv/recurrent state 和 GPU 历史/长度均与相同缓存条件下的普通批量基准逐元素精确一致；prefix 实际命中 3 次，覆盖候选 K_i=0/1/2/3/4 混批和跨页。记录位于 `logs/validate/spec_batch_warm_prefix_final.{json,log}`。

补充 128-token 长输出、长 prefix、普通 decode 图开启的最终验收：batch=3/8 × prefix 开/关，共 **4 个场景、496 个批次、2730 个请求端点通过**；最终词元、KV、conv/recurrent state 与同缓存条件的普通基准全部精确一致，prefix 实际命中 3 次。记录位于 `logs/validate/spec_batch_128_final.{json,log}`。与上述 64-token 扫描合计 8 个场景、743 个批次、4197 个请求端点、6 次实际 prefix 命中均通过。中途因执行环境切换而未完成的 `spec_batch_anchor_guard_final.json` 保持 `completed=false`，不计入通过数量。

prefix 冷/热条件单独记录：在 batch=4 的长 prefix 场景中，**关闭投机的普通解码**第一次冷缓存生成与后续热缓存生成也出现部分词元与状态差异。最初冷基准对热投机的结果保留在 `logs/validate/spec_batch_long_prefix_final.{json,log}`，标记失败；最终对照先温热普通基准和投机双方的缓存，并保留 `baseline_cold_warm_matches`、`baseline_cold_warm_state_failures`、双方的实际 prefix 命中数。温热条件下通过不表示已解决普通模型的跨批次/缓存数值不变性；这仍是后续系统验收问题。

显式 native 批处理额外运行 batch=4、32-token 输出、长 prefix 开/关，共 22 个批次、79 个请求端点：所有同路径恢复检查通过，15 个批次包含候选拒绝；实际未运行参考验证。两场景的词元均与普通基准相同，但最终 KV/conv/recurrent state 不同，因此严格验收 `passed=false`、命令退出 1，记录位于 `logs/validate/spec_batch_native_final.{json,log}`。不能将这次短输出一致作为 native 已修复的证据。

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_batch.py --output-tokens 64 --batch-sizes 4 8 \
  --modes packed_guarded --decode-graphs --long-prefix \
  --json-out logs/validate/spec_batch_warm_prefix_final.json
```

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

显式配置 `verification_mode="packed"` 仅使用主模型批量验证的 argmax 接受/拒绝候选，不进行逐词元参考检查、不克隆 KV 用于对照、不因普通路径的浮点数值差异回退。全接受复制试算最终 GDN 状态；部分拒绝或截断重放有效输入。仅资源不足时恢复私有状态并回退到逐词元路径。2026-09-28 起它是实验性的显式选项，不再是开启后的默认值。

`verification_mode="sequential"` 提供正确性参考。保守配置 `verification_mode="packed_guarded"`（开启后的默认值）：单次 eager 因果前向输入 K+1 个词元，投影所有预测行，随后使用相同入口状态运行逐词元参考检查。逐词元试算覆盖 native 试算 KV 尾部，只有精确一致时使用 native 预测；存在 argmax、GDN 或 KV 差异时使用参考结果和状态。显存不足也恢复私有状态后使用逐词元路径。两种路径都不发布候选历史。

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

`benchmarks/bench_spec_decode.py` 使用真实 n-gram、没有预知输出的草稿、没有状态快照，测量单请求离线纯解码及整段生成。默认对照启用 decode 图的普通路径，packed 验证使用 eager 路径；`--enforce-eager` 可改为两者均 eager。默认 509 个 prompt token、128 个输出 token、K=4，每种负载/模式预热 1 次、测量 3 次，记录真实接受率、回退、重放、峰值显存、基准词元匹配与耗时。2026-09-28 起默认仅测 baseline/packed，按轮交错并轮换先后顺序，保存每轮配对时间比；`--modes baseline sequential packed` 可保留旧的三模式对照。结果写入 JSON，不作为服务 TTFT/P95 指标。

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

## 扩展验收与交错性能对照（2026-09-28）

新增 `benchmarks/spec_workloads.py` 和 `benchmarks/validate_spec_natural.py`。输入包括英文解释、中文解释、仓库 `input_prep.py` 的真实代码、包含不同观察记录的长输入、低匹配的创作提示；自然提示只截断，不复制填充。长输入按上限截断到 2048 token。

验证使用实际 n-gram，不使用预知输出草稿；记录正式 GPU 历史和已计算端点、每轮私有状态恢复、非活动 slot、最终有效 KV/conv/recurrent state，以及每个预测位置的 argmax 和前两名分数差。`topk` 的并列次序不代表 argmax，诊断必须报告实际 argmax；未提交候选和分块预填充中 emit=0 的预测不能当作实际输出分歧。完整的已提交预测轨迹必须与实际输出逐词元、逐位置对齐。near-tie 仅用于定位，不豁免词元失败。中途异常的 JSON 标为未完成，词元失败保存完整证据后返回非零退出码。

初始 K=4、128-token 输出、prefix 开/关共 20 个对照场景：10 个 sequential 全部一致；10 个正常 packed 中 6 个出现词元分歧，分别是中文、仓库代码、低匹配提示及各自的 prefix 开启版本。失败位置分别为输出的第 25、19、101 个词元。恢复检查未失败，说明恢复正确不能保证跨执行路径的预测完全一致。初始 JSON 位于 `logs/validate/spec_natural_initial.json`；首轮 topk 诊断的并列顺序已修正，最终定位结果以修正后重跑为准。

保守路径完整验收：5 种输入 × prefix 开/关 × K=1/2/4/8 × sequential/packed_guarded，共 **80 个场景、3684 个提交端点全部通过**。每个场景输出 256 token；最终已确认 GPU token、有效 KV、conv/recurrent state 与普通逐词元基准逐元素精确一致。guarded 检查发现 40 轮 native 预测不同，已通过参考路径处理。结果位于 `logs/validate/spec_natural_conservative.{json,log}`。这验证了保守回退，不代表 native packed 已修复，也不代表加速。

修正并列预测及非输出预填充行的诊断后，再运行 K=4、128-token 输出的 sequential/packed_guarded 对照：**20 个场景、258 个提交端点通过**，已提交预测轨迹与真实输出逐词元、逐位置一致。结果位于 `logs/validate/spec_natural_trace_final.{json,log}`。

修正预测诊断后，native packed 扫描 5 种输入 × prefix 开/关 × K=1/2/4/8、每场景输出 128 token，共 **40 个场景，24 个词元失败**；643 个提交端点的同分块恢复检查全部通过。中文在第 25 个输出词元、代码在第 19 个、低匹配提示在第 70 或 101 个首次分歧。24 个首个分歧都来自后续普通解码，不是当前候选验证行，表明先前提交的不同数值状态会影响后续预测；当前轮低分差检查不能解决这类问题。该扫描实际命中 prefix 快照 8 次，native 没有回退。JSON 标记 `passed=false`，命令退出码为 1，记录位于 `logs/validate/spec_natural_packed.{json,log}`。

```bash
# 严格模式：自然输入、长序列、K 扫描及 prefix 开关
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_natural.py --output-tokens 256 \
  --draft-sweep 1 2 4 8 --modes sequential packed_guarded \
  --json-out logs/validate/spec_natural_conservative.json

# 正常 packed：同一严格门槛，真实分歧会保存 JSON 并返回非零
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_natural.py --output-tokens 128 \
  --draft-sweep 1 2 4 8 --modes packed \
  --json-out logs/validate/spec_natural_packed.json

# 自然负载性能；每个轮次配对，先后顺序轮换
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/bench_spec_decode.py --suite natural --prompt-tokens 2048 \
  --output-tokens 128 --draft-sweep 1 2 4 8 --repeats 6 \
  --json-out logs/bench/spec_natural_interleaved.json
```

性能 JSON 在输出不同的配置上将 `speedup_comparable` 设为 false、`decode_speedup` 和 `median_paired_decode_speedup` 设为 null；原始时间比仍保留作诊断。`--require-token-match` 可要求性能脚本在输出不同后返回非零。速度比只能引用输出一致的配置，不能把分歧后的不同生成任务视为相同工作量的加速证明。

交错性能实测已完成：RTX 3060 Ti、Qwen3.5-0.8B BF16，普通解码启用 decode 图，native packed 使用 eager；prefix 关闭。5 种输入 × K=1/2/4/8 × baseline/packed，各预热 1 次、计时 6 次，共 240 次计时生成，每次输出 128 token。每轮 baseline/packed 成对运行，先后顺序交替，结果位于 `logs/bench/spec_natural_interleaved.{json,log}`。

| 输出一致的样例 | K=1 | K=2 | K=4 | K=8 |
|---|---:|---:|---:|---:|
| 英文自然提示：配对解码速度比中位数 | 0.858× | 0.845× | 0.859× | 0.860× |
| 长观察记录：配对解码速度比中位数 | 0.787× | 1.084× | 1.637× | 2.398× |

20 个 native 配置中仅上述 8 个配置的所有计时输出与基准一致；其余 12 个（中文、仓库代码、低匹配提示的各 K）全部存在分歧，速度比标为不可比较。长观察记录在 K=1 时接受率约 93.85% 仍减速，K=8 时接受率 75% 却有收益；后续自适应策略需要综合每轮输出数、验证和恢复开销，不能只用接受率做决策。这些是固定小样例的离线测量，不代表自然流量总体收益，也不代表保守默认模式有加速。

最终全量回归：**97 项测试全部通过**，包括默认配置在 native 词元及状态分歧时使用参考结果、显式 native 模式不额外运行参考路径、BF16 并列 argmax 诊断和输出不同的配置不报告有效加速比。测试日志位于 `/tmp/nano_qwen_spec_final_20260928.log`。

不传 `--verification-mode` 再运行原 GPU 验收，确认命令默认随配置选择 `packed_guarded`：**36 个场景、84 个状态端点通过**，词元和有效 KV/GDN 状态与普通解码精确一致，prefix 快照实际命中 18 次。EOS、长度、上下文结束及跨页仍通过，记录位于 `logs/validate/spec_default_guarded_final.{json,log}`。

补充启用普通 decode 图的实际生成对照：5 种输入、K=4、128-token 输出，baseline/packed_guarded 各预热一次并运行一次，全部输出一致，`--require-token-match` 通过。该运行是图路径词元回归，不用其单次计时宣称性能，记录位于 `logs/bench/spec_default_graph_fidelity.{json,log}`。显式 native 的预测轨迹也在长观察记录、prefix 开/关两个场景通过完整对齐检查，记录位于 `logs/validate/spec_native_trace_final.json`。

## 后续工作

正常 packed 的自然输入分歧已复现；P2 首版采用保守批量锚点路径；下一步继续补齐系统验收，并修复 native 数值路径后再提升多词元默认行为。不能仅在当前轮 top-2 分数接近时回退：此前提交的 packed 状态也可能影响后续普通解码。恢复位置、已确认历史、页引用与结束条件仍严格检查。

P2 首版变长批处理、GPU 接受/提交及独立异步输出句柄已接入；继续补齐复杂到达、取消/抢占、容量压力等系统验收。多请求保守模式维持每请求一步输出；native 数值问题和 GPU 全程恢复尚未解决。P3–P7 的性能优化、随机拒绝采样和真实模型草稿均尚未实现。MTP/EAGLE/DFlash 没有创建占位模型支持。
