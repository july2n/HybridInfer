# vLLM 投机解码对齐审计与六种模型草稿适配

审计日期：2026-09-28。最终目标为 **EAGLE-3、P-EAGLE、DFlash、DFlash2、DSpark、MTP**；n-gram 仅用于验证公共链路。当前不能声明这六种方法已经支持，也不能声明 native target 执行已经与 vLLM 全部对齐。

## 1. 固定参考与对齐的定义

源码参考：本地 `/home/lang/workspace/vllm` 的 `a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`。同时检查 V1 `gpu_model_runner.py` 和新 `worker/gpu` runner；仅检查旧 runner 的 n-gram 路径不足以覆盖六种模型草稿。在线官方 Speculators 文档用于解释模型契约，执行与路由以固定源码为准。

环境中已安装的 vLLM 是 **0.19.0**，不是上述源码构建。完整模型参考实验须记录这个版本差异，不能将它标记为固定提交的端到端验收。

对齐分为独立的门槛：

1. **相同张量的协议对齐**：预测行、变长 offsets、连续接受前缀、correction/bonus、有效输出与端点应与实际 vLLM 函数逐元素相同。
2. **本引擎状态恢复正确性**：与相同输入、相同执行路径的参考比较有效 KV/conv/recurrent 和 CPU/GPU 长度；拒绝尾部不可见，不发布缓存。
3. **完整模型执行对照**：固定模型、精度、prompt IDs、batch/缓存条件，比较普通/投机生成和 vLLM 参考；数值差异与协议差异分别记录。本仓库已有的严格贪心 token 门槛继续保留。
4. **真实草稿后端验收**：实际权重加载、目标特征契约、草稿缓存恢复、采样分布、拒绝/终止与性能都必须验证。接口或方法名通过不等于后端已支持。

随机采样的正确性门槛是目标条件分布和随机计数域正确；不能将非投机与投机同 seed 的逐 token 相同视为拒绝采样的算法保证。vLLM 的质量评估也不能替代本引擎的状态端点验收。

## 2. 新增可执行差分检查

`benchmarks/validate_vllm_spec_alignment.py` 从固定提交原文件加载未修改的 AST 定义，保留原文件名供 Triton JIT 读取源码，不重写参考接受算法。仅显式提供依赖及 H2D 搬运；没有构建或运行完整 vLLM engine。

实际执行的参考包括：

- `GPUModelRunner._calc_spec_decode_metadata` 和 `_get_cumsum_and_arange`。
- 旧 V1 `rejection_greedy_sample_kernel`。
- 新 GPU runner `rejection_sampler_utils.rejection_sample` 及其实际 Triton/Gumbel 依赖。

检查覆盖 K=0…8、变长混批、额外 prefill 行、所有接受长度、拒绝后后续候选再次匹配、bonus、BF16 并列 argmax、跨词表 tile、request-state slot 重排。EOS/输出预算在 vLLM 接受结果之后截断，不能假装 vLLM 接受 kernel 自身处理了结束条件。

本次结果：CPU **104** 组布局、CUDA **104** 组布局、新旧接受路径 **112** 个批次/**931** 个请求全部通过。证据：`logs/validate/vllm_spec_alignment_20260928.{json,log}`；JSON 保存固定提交、7 个参考文件的 SHA256、环境和测试范围。

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_vllm_spec_alignment.py --trials 100 \
  --json-out logs/validate/vllm_spec_alignment_20260928.json
```

结论仅限当前贪心元数据/接受协议，不覆盖随机 sampler、完整 scheduler、目标模型数值或六种模型草稿。

## 3. 公共链路仍未对齐的部分

| 环节 | 固定 vLLM 源码行为 | HybridInfer 当前行为 | 下一步 |
|---|---|---|---|
| 预测行与贪心接受 | K_i+1 预测行；首拒绝 correction；全接受 bonus | 直接差分已通过 | 保留该差分门槛，扩展真实 logits/随机分布 |
| 新旧 cumulative layout | 旧 metadata 是 B 项累计结束位置；新 GPU runner 的 `cu_num_logits` 为 B+1、含起始 0 | 当前内部使用旧式 B 项累计结束位置 | 对新 runner 参考显式加 0，禁止混用坐标系 |
| GDN 多词元验证 | spec 分支使用 `causal_conv1d_update` 与 `fused_sigmoid_gating_delta_rule_update`，提供按位置的状态索引及接受数 | `set_context(True)` 后使用 prefill conv/chunk scan | 建立专用短链 recurrent 验证路径，与 prefill 分离 |
| GDN 拒绝恢复 | 按已接受端点选择 recurrent state；conv 同步接受偏移，跨块迁移修正偏移 | 私有最终状态；部分接受取回长度后重放 | 逐位置/分段私有状态 + GPU 端点选择；不要机械复制 vLLM 物理布局 |
| 多请求默认执行 | 正常验证一次 target 多词元前向，提交连续接受段 | guarded 额外试算后只提交普通 batch 形状的一步锚点 | native 通过门槛后再启用多词元默认行为 |
| 草稿输入与输出 | 设备上的 hidden/aux features、sampled/rejected、last token、slot mapping 与草稿概率缓存 | CPU token tuple，只有 n-gram proposer | 建立设备批次后端协议与 target features 输出 |
| 随机采样 | target sampling params、点质量或真实 q、概率比接受、residual correction；draft noise 与 target 分离 | temperature 非零整批回退；只有贪心接受 | 公共拒绝 sampler 独立验收，再接真实后端 |
| 资源预算 | target 验证页之外，另计草稿 query slots/KV groups | 只按 target trial endpoint 预留 | 区分 target KV、draft context KV、draft query slots |
| 草稿缓存生命周期 | 按 sampled/rejected 修正草稿缓存及特征位置 | 没有模型草稿缓存 | commit/abort/preempt/finish 钩子及独立所有权 |

源码锚点：

- [Qwen GDN spec 更新](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1250)。
- [GDN metadata 与状态索引](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/attention/backends/gdn_attn.py)。
- [混合状态跨块迁移](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/mamba_utils.py#L519)。
- [BaseSpeculator.propose](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu/spec_decode/speculator.py#L68)。
- [新 GPU runner 的拒绝采样](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py#L998)。

这是源码路径差异证据，并非已经证明“所有输出差异只由 chunk scan 导致”。Dense GEMM 的行数、attention backend 和 GDN 运算/舍入次序都可能贡献差异。采用 recurrent 验证也不自动保证所有 BF16 贪心输出与单步 decode 完全相同。

部分接受还有明确的执行语义差异：`batch_execution.py` 取回输出长度后，对缩短的输入/请求子批重新执行整个 target，不仅重算 GDN，也重算 dense/attention 并覆盖有效 KV。vLLM 则保留原验证前向的有效 KV，并按接受位置读取/迁移 recurrent/conv 状态。因此当前“同分块重放参考通过”证明的是重放路径自身正确，不能证明提交状态就是原 K+1 验证前向在接受端点的状态。新的状态选择验收必须独立检查**原验证前向的逐位置端点**，并确认拒绝尾部特征不进入下一轮草稿。

## 4. 六种方法的独立适配契约

| 方法 | 固定 vLLM 路由 | 特征/草稿契约 | 不可省略的适配 |
|---|---|---|---|
| EAGLE-3 | `method=eagle3`，自回归 proposer | checkpoint 指定的目标辅助层特征、token embedding、draft-to-target 词表映射；逐步草稿 | 特征采集点/层序/残差与归一化位置；特征和 token 一位偏移；draft KV rollback |
| P-EAGLE | `eagle3` + `parallel_drafting=True` | 经过并行训练的 EAGLE-3 checkpoint，一次产生多深度候选 | 独立的并行输入、mask token 和位置；通常额外 K-1 query slots；不能把普通 EAGLE-3 循环改为并行就宣称支持 |
| DFlash | `method=dflash` → `DFlashSpeculator` | 目标特征先构建 context KV；anchor + K mask queries，一次 block forward | 草稿专用因果/非因果 attention 分组、context/query KV、mask 与 position；通常额外 K slots；target 验证始终因果 |
| DFlash2 | `dflash` + `DFlash2DraftModel` → `DFlash2Speculator` | DFlash backbone 加分组局部动态卷积及 top-K predecessor-conditioned selector | 卷积不能跨 anchor block；selector 实际路径及条件分布缓存；稀疏候选 logits 清理；当前格式要求完整目标词表 |
| DSpark | `method=dspark` → `DSparkSpeculator` | 并行 backbone + 顺序 Markov head；可选 confidence head | `sample_from_anchor` 按 checkpoint 解析：K 或 K+1 query；真实 Markov 修正后的 q、词表映射；confidence 只能决定预算，不能替代接受概率 |
| MTP | `method=mtp` → `MTPSpeculator`；模型专属 MTP 架构 | 目标 hidden + 下一输入 token embedding；独立 norm、concat/FC、预测层递推；embedding/head 可共享 | 真实 MTP 权重完整加载、特征位置、spec_step_idx、独立 draft KV；Qwen3.5 MTP 预测层为 full attention |

路由以 [init_speculator](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu/spec_decode/__init__.py) 和 [草稿 slot 预算](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/config/speculative.py#L1883) 为准。方法同一名称不表示模型格式可以互换；DSpark query 布局存在 checkpoint 差异。

首轮支持范围为线性候选链。vLLM EAGLE 的可选树状候选不在本次差分范围内；若随后要求树验证，必须另加 target 树状因果掩码、分支 GDN 状态和树形位置映射，不能把树节点扁平化成普通线性历史。词表映射也必须按 checkpoint 核对是绝对目标 ID 还是 draft ID 的偏移量。

官方模型说明：[EAGLE-3](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/eagle3.md)、[P-EAGLE](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/peagle.md)、[DFlash](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash.md)、[DFlash2](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dflash2.md)、[DSpark](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/dspark.md)、[MTP](https://github.com/vllm-project/speculators/blob/main/docs/user_guide/algorithms/mtp.md)。

### 本地 MTP 权重核查

`models/Qwen3.5-0.8B/config.json` 设置 `mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`。实际 safetensors 存在 **15 个 mtp 张量**，包括 `mtp.fc.weight [1024,2048]`、完整 full-attention Q/K/V/O 与 Q/K norm、MLP、输入/输出及两路 pre-FC norm。不能再将“可能没有 MTP 权重”作为未经检查的前提。

目前 `utils/loader.py` 仍显式跳过这些权重，目标 forward 只返回最终 hidden，尚无可用 MTP 模型/草稿缓存。[vLLM Qwen3.5 MTP](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/model_executor/models/qwen3_5_mtp.py) 可作为第一项真实后端对照。权重存在不等于本引擎已能执行。

15 个 MTP 张量共 **20,452,864** 个参数，BF16 约 **39.01 MiB**，不含共享 embedding/head 与草稿 KV。相较下载额外模型，它适合作为本地首个真实草稿验收对象。

### 状态和特征接口的具体约束

- vLLM 辅助特征在指定边界采集 `hidden_states + residual`，不是随意选某层归一化后的输出；默认层只是无配置时的 fallback，实际层序以 checkpoint 为准。应显式实现所需层的采集及元数据，并用真实 vLLM 特征函数检查 residual/层索引契约。
- `DraftProposal` 的概率语义至少区分确定草稿（点质量）和概率草稿（实际条件 q）。DFlash2 缓存 selector 选择路径的 top-K scores；DSpark 缓存加入 Markov bias 后的 logits；不能缓存原始 unary logits 后用于拒绝采样。
- 增加按请求的 draft KV 和特征缓存端点，commit 同步实际 sampled/rejected；abort、抢占和结束不得留下未来候选特征。目标特征输出与草稿查询位置必须分别定义。
- target 输入始终为 `[u,d1,...,dK]`，与草稿侧 K 或 K+1 query 的布局分开。若实际输出数为 m，提交输入仅为 `[u,d1,...,d_(m-1)]`，computed 推进 C+m，最后输出仍未计算。用于逐位置状态选择的本地零基端点为 m-1；EOS/预算截断后必须从真实 m 计算，不能直接套用草稿接受数 a 或 vLLM 某字段名称。
- 新状态选择不能不计显存地保存所有端点：本地模型 18 个 GDN 层，每请求每端点 FP32 recurrent state 为 **18 MiB**。B=8、K=4 的 K+1 端点为 **720 MiB**，K=8 为 **1296 MiB**，还未包含 conv、入口备份、KV、特征和草稿模型。RTX 3060 Ti 的 8 GiB 下应先比较逐位置、分段保存与 GPU 重放方案。

## 5. 实施顺序调整

1. **公共 target 多词元执行与状态选择**：专用 recurrent 验证；同路径有效 KV/GDN 端点检查；拒绝/终止/跨页/slot 重排；真实自然输入严格门槛。并行补动态到达、容量压力和取消/抢占系统验收。
2. **设备批次 proposer 协议与 target features**：按 checkpoint 导出最终/辅助 hidden，显式特征位置、token/feature shift、request slots、sampled/rejected、draft 分布语义、词表映射和 draft 缓存生命周期。不要为每个方法复制 target verifier。
3. **共享随机拒绝采样**：确定草稿采用点质量还是概率采样；真实 q 按已经实现的候选条件路径缓存；温度/约束与 target 一致。小词表统计、slot 重排和未提交随机计数隔离先通过。
4. **MTP**：使用已存在的 Qwen3.5-0.8B 权重进行真实加载与 vLLM 参考实验；验收特征/缓存偏移、全接受/部分拒绝、EOS/预算与实际收益。
5. **EAGLE-3 → P-EAGLE**：找到与目标模型兼容的 checkpoint，先完成自回归特征和缓存契约，再接并行输入/预算；没有兼容权重时记录模型依赖，不用 mock 宣称完成。
6. **DFlash → DFlash2、DSpark**：共用 context KV/块级输入框架，分别实现 selector 或 Markov/confidence，分别验收真实 q 与 checkpoint 布局。
7. **性能优化与图执行**：对已经通过的真实后端分项测量 draft/target/state/commit；再做 K 自适应、图捕获和低收益回退。n-gram 不再作为项目最终收益验收对象。

## 6. 本次真实模型重测

全量 GPU 单元回归 **112 项通过，无跳过**，日志 `/tmp/nano_qwen_alignment_tests_20260928.log`。

Native 自然输入：5 种输入 × prefix 开/关 × K=1/2/4/8，每场景 128 输出 token，共 **40 场景、24 个 token 失败**。**643 个同路径恢复端点均通过**；24 个首个分歧都来自后续 `ordinary` decode。结果：`logs/validate/spec_alignment_native_20260928.{json,log}`，`completed=true, passed=false`，进程退出 1。不能用恢复通过替代完整预测通过。

保守自然输入：5 种输入 × prefix 开/关 × K=1/2/4/8 × sequential/packed_guarded，每场景 256 输出 token，共 **80 场景、3684 个提交端点全部通过**。最终 token、有效 KV/conv/recurrent 与普通参考逐元素相同，实际 prefix 命中 16 次，guarded 处理 40 次 native 预测差异。证据：`logs/validate/spec_alignment_conservative_20260928.{json,log}`，`completed=true, passed=true`，退出 0。

批量长输出：batch=3/8 × prefix 开/关、普通 decode 图开启、509-token 长 prompt 与每请求约 128 输出 token，共 **4 场景、496 个批次、2730 个请求端点通过**。相同缓存条件下最终 token、有效 KV/conv/recurrent、GPU 历史与长度均与普通 batch 基准逐元素相同；实际 prefix 命中 3 次。证据：`logs/validate/spec_alignment_batch_20260928.{json,log}`，`completed=true, passed=true`，退出 0。

普通 baseline 冷/热自身差异仍存在：本次 batch=3、prefix 开启时首请求 cold/warm token 和状态不同，另外两个请求一致。双方温热后对照通过不表示该系统数值问题已修复；JSON 保留 `baseline_cold_warm_matches` 和状态差异。

安装版 vLLM 普通模型参考已完成：vLLM 0.19.0、相同 prompt IDs、BF16、FP32 recurrent、eager、prefix 关闭、target token 预算 2048；5 种输入 × batch=1/4 共 **10 个场景、25 个请求**均成功生成 128 token。batch=1 对 HybridInfer 普通参考只有中文和长观察记录一致；英文、仓库代码、低匹配提示首次分歧为输出第 **33、19、70** 个 token（JSON 使用零基索引 32/18/69）。记录 `completed=true, execution_passed=true, hybrid_token_passed=false, passed=false`，退出 1。证据：`logs/validate/vllm_model_baseline_20260928.{json,log}`。batch=4 没有与 HybridInfer batch=1 参考混比。

这说明跨引擎普通 target 数值/词元一致性也未通过，不能将 native 的全部失败仅归因于投机接口。该完整参考使用安装版旧 runner，不是固定提交的新 runner 构建；还需要逐层定位具体数值来源。

安装版 vLLM 真实 MTP：确认解析为 `method=mtp, num_speculative_tokens=4, draft_architectures=[Qwen3_5MTP]`；加载了本地 MTP 权重并共享 target embedding/head。5 种输入 × batch=1/4 共 **10 场景、25 个请求**均生成 128 token，`execution_passed=true`。

严格与同版本、同参数 vLLM 普通基准对照：仅长观察记录的 batch=1/4 两个场景一致，即 **5/25 请求相同、20/25 不同**。batch=1 英文/中文/代码/低匹配首次分歧零基索引为 **52/79/116/32**；batch=4 每请求索引完整保存。对 HybridInfer batch=1 参考仅长观察记录一致（1/5）。证据：`logs/validate/vllm_model_mtp_20260928.{json,log}`，`completed=true, passed=false`，退出 1。

这个结果不能推广为固定提交新 runner 的结果，也不能证明差异必然只来自某个 kernel。它证明本环境中 vLLM 0.19.0 的真实 MTP 执行也未通过逐 token 完全相同的门槛。因此必须区分“同 logits 的接受协议精确对齐”“目标状态/分布正确”“不同浮点执行路径输出完全相同”。不更改已有严格门槛、不将这些差异豁免为通过，也不以 vLLM 的数值差异解释或掩盖本引擎的恢复错误。

对应命令：

```bash
PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_natural.py --output-tokens 256 \
  --draft-sweep 1 2 4 8 --modes sequential packed_guarded \
  --json-out logs/validate/spec_alignment_conservative_20260928.json

PYTHONPATH=src:.runtime-deps /home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_spec_batch.py --output-tokens 128 \
  --batch-sizes 3 8 --modes packed_guarded --decode-graphs --long-prefix \
  --json-out logs/validate/spec_alignment_batch_20260928.json
```

`benchmarks/validate_vllm_model_reference.py` 为独立的完整引擎参考：使用同一组 prompt IDs，baseline/MTP 分进程执行，记录实际 vLLM 版本及 batch=1/4 的 token 对照。该脚本不检查 vLLM 内部逐层状态，也不代替本仓库固定源码 kernel 差分或六方法的 checkpoint 验收。

```bash
/home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_vllm_model_reference.py --mode baseline \
  --fixtures logs/validate/spec_alignment_native_20260928.json \
  --hybrid-reference logs/validate/spec_alignment_conservative_20260928.json \
  --json-out logs/validate/vllm_model_baseline_20260928.json

/home/lang/anaconda3/envs/vllm_env/bin/python \
  benchmarks/validate_vllm_model_reference.py --mode mtp --draft-tokens 4 \
  --fixtures logs/validate/spec_alignment_native_20260928.json \
  --baseline logs/validate/vllm_model_baseline_20260928.json \
  --hybrid-reference logs/validate/spec_alignment_conservative_20260928.json \
  --json-out logs/validate/vllm_model_mtp_20260928.json
```

baseline 命令因跨引擎 token 失败返回 1，但 `completed=true, execution_passed=true` 的输出仍是有效 vLLM 普通参考；MTP 脚本会检查参考的执行完成情况、模型/精度/版本/预算及 prompt IDs。两个命令应分别运行并检查 JSON，不用 `&&` 把预期的严格失败当作未执行后续实验。
