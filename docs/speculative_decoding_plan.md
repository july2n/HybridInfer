# 投机解码实施计划

本文定义实施范围和验收门槛；实际交付与验证结果见[实施记录](speculative_decoding_progress.md)。目前 P0 与单请求 n-gram 贪心执行链路已实现，正常 packed 路径的扩展词元一致性验收仍未通过，P2 已接入首版变长批量执行、GPU 接受/提交与异步输出句柄，完整系统验收仍在补齐。第一阶段以 n-gram 草稿和贪心验证建立可验证的执行链路，最终支持 MTP、EAGLE/EAGLE3、DFlash 等草稿方法。默认关闭，关闭后保留现有执行行为。

## 1. 目标与范围

建立“生成草稿 → 主模型批量验证 → 接受/拒绝 → 状态提交”的通用机制。草稿方法可以替换，主模型验证、历史状态管理、调度和输出协议保持共用。

第一版范围：Qwen3.5 Dense、单 GPU、线性候选链、temperature=0、CPU n-gram 查找、普通执行路径。无匹配、预算不足或资源不足时回退到单词元解码。初始正确性阶段不混合预填充与投机验证，不捕获新的完整验证图。

后续分阶段加入变长批量验证、异步执行、前缀缓存协作、图执行、随机拒绝采样及模型草稿适配。树状候选需要独立的注意力掩码和 GDN 分支状态设计，不能直接沿用线性候选协议。

## 2. 实施前代码与需要改变的契约

下表保留设计时的基线和改动目标；已经实现的部分及尚未通过的验收见实施记录。

| 位置 | 当前实现 | 所需变化 |
|---|---|---|
| `engine/sequence.py` | 每次 `append_token`，CPU 维护已确认历史 | 批量追加已提交词元；候选不进入正式历史 |
| `scheduler/scheduler.py` | 解码固定调度 1 个输入词元 | 根据草稿长度预留验证预算与 KV 容量，提交后推进真实长度 |
| `engine/model_runner.py` | 普通预填充只投影每请求最后一行；执行后立即 `advance` | 为验证选择全部有效预测行；试算期间不推进正式长度、不保存 prefix 快照 |
| `engine/input_prep.py`、`request_state.py` | GPU 常驻历史、长度；采样结果单词元提交 | 独立候选输入缓冲区和批量提交内核，拒绝词元不写入正式历史 |
| `engine/async_output.py`、`llm_engine.py` | 每请求返回一个词元，结束判断按单词元执行 | 返回变长词元段、有效长度与接受统计，逐词元处理 EOS 和长度截断 |
| `engine/block_manager.py` | 单步追加、完整块哈希发布 | 多词元容量预留、试算块回收，只有已确认且已计算的完整块可发布 |
| `utils/context.py`、`engine/cuda_graph.py` | 有 `spec_decode` 描述预留，但运行路由仍落入单步 decode 图 | 明确验证模式使用多词元因果路径，禁止误用单词元图 |
| `utils/loader.py` | 当前跳过 MTP 权重 | 接入 MTP 时增加显式模型分支、独立加载与完整性检查 |

`BatchDescriptor` 中的模式字符串不是投机解码实现。实际模型形状、采样行、状态更新和图路由都需要同步修改。

## 3. 通用架构

已建立 `src/hybridinfer/spec_decode/`，按以下职责继续扩展：

- `config.py`：`SpeculativeConfig`，包含方法、候选数上限、n-gram 范围、草稿模型路径、验证模式、状态恢复策略及显存预算。
- `interfaces.py`：定义 `DraftContext`、`DraftProposal`、`VerificationPlan`、`VerificationResult` 和后端能力声明。
- `ngram.py`：仅从已确认 token 历史查找候选。
- `verifier.py`：CPU 正确性参考；`metadata.py`：变长批次与两层行索引；`batch_verifier.py`：GPU 连续接受前缀、补偿/额外词元和结束条件。
- `batch_execution.py`：一次打包 target 前向、私有状态与部分接受恢复；`commit.py`：GPU 有效输出批量提交；`async_output.py`：独立变长输出缓冲区与完成事件。
- `state.py`：验证状态事务、KV 预留和 GDN 状态恢复。
- `metrics.py`：候选数、接受数、回退原因、各阶段时间与显存。
- 后续添加 `mtp.py`、`eagle.py`、`dflash.py` 及各自模型实现。

公共接口的概念定义如下，具体字段在实现时依现有批处理布局确定：

```python
proposal = proposer.propose(draft_context)
plan = verifier.prepare(proposal, committed_state)
trial = runner.verify(plan)
result = verifier.accept(trial, proposal, sampling_params)
state_manager.commit(plan, result)
proposer.on_commit(result)
```

`DraftContext` 包含请求身份、已确认历史、逻辑位置及按需提供的目标模型特征。`DraftProposal` 使用扁平候选 token、请求 offsets/lengths 和可选草稿分布信息；不能假定所有请求候选数相同，也不能只存一个置信度标量代替完整概率契约。

`VerificationResult` 区分草稿接受数、实际输出词元数、实际计算词元数、终止标记及已确认位置。异步输出使用 `[B, max(K_i)+1]` 词元缓冲区与 `[B]` 有效长度，或等价打包布局，每个输出句柄独立持有缓冲区和完成事件。

草稿后端声明是否需要目标隐藏层、哪些层及归一化位置、词表映射、草稿状态、随机采样支持和候选拓扑。不要求所有方法共享同一种内部前向过程。

## 4. n-gram 草稿规则

建议初始参数：候选数上限 K=4，匹配长度范围 2–8；支持配置，后续测量 K=1/2/4/8。

从最长匹配长度向下搜索，将当前历史末尾的 n 个 token 与更早位置比较；同长度多个匹配时选择最近的、存在后续词元的匹配，复制该历史出现后的最多 K 个 token。不使用本次尚未确认的候选扩展搜索历史。

只返回已存在历史中的后续 token，不循环构造无限候选。不足 K 时返回短候选；没有匹配时返回空草稿。按输出剩余长度、模型上下文余量和调度预算裁剪候选。CPU 查找先采用简单独立实现，性能验证后再考虑增量索引或 GPU 查找。

## 5. target model 如何验证一次提出的 K 个候选

### 5.1 vLLM 源码核对基准

本节于 2026-09-28 核对本地 `/home/lang/workspace/vllm`，固定版本为 `a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`，以下引用指向该提交。核对的是 V1 `GPUModelRunner` 的线性候选验证链路，其他 runner、树状候选和未来版本需另行核对。

| 环节 | 源码入口 | 核对结论 |
|---|---|---|
| 行布局与候选对齐 | [`GPUModelRunner._calc_spec_decode_metadata`](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu_model_runner.py#L2801) | 每请求选择最后 K_i+1 个预测位置，候选 token 对应其预测行的下一输入位置 |
| 元数据契约 | [`SpecDecodeMetadata`](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/spec_decode/metadata.py#L9) | 分别保存候选累计长度、采样累计长度、hidden-state 行索引及 target/bonus logits 行索引 |
| 预测投影 | [`sample_hidden_states` / `compute_logits`](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/gpu_model_runner.py#L4461) | 从本次前向的 hidden states 选择有效行后投影，不能沿用普通 prefill 只投影最后一行的逻辑 |
| 接受/拒绝 | [`RejectionSampler.forward` 与 GPU kernels](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/sample/rejection_sampler.py#L140) | 分离 target 与 bonus 行；GPU 上按请求求最长接受前缀；拒绝后不输出后续候选或 bonus |
| 长度修正 | [`Scheduler.update_from_output`](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/core/sched/scheduler.py#L2048) | 根据实际输出扣除拒绝候选对应的试算长度，并处理异步占位长度与过期结果 |

### 5.2 一次因果前向，K+1 行预测

正常解码入口满足：已计算长度为 C，正式 token 历史长度为 C+1；最后一个已确认 token 是尚未计算的锚点 u。草稿为 d1…dK，历史为 H（包含 u）。target 本轮输入 `[u, d1, ..., dK]`，位置为 C…C+K，复用此前 C 个 token 的缓存，通过一次多词元因果前向得到 K+1 行 hidden states 和 logits。

每一行只能看到已缓存前缀及该行以前（含该行）的输入。于是验证 d_i 的目标分布是 `p_i = P_target(. | H, d1, ..., d_(i-1))`，最后一行为 `p_(K+1) = P_target(. | H, d1, ..., dK)`。target 不需要先确认 d1 才计算 d2 的验证分布；它用完整草稿做 teacher forcing，在同一次前向中计算全部条件分布，随后由接受逻辑判断哪些条件前缀有效。注意力使用因果掩码，GDN 扫描也必须按因果递推；一次前向不意味着递归状态在所有位置独立计算。

以 K=3 为例：

| 输入行/位置 | 本行输入 | 本行 logits 的条件 | 用途 |
|---|---|---|---|
| 0 / C | u | H | 验证 d1 |
| 1 / C+1 | d1 | H, d1 | 验证 d2 |
| 2 / C+2 | d2 | H, d1, d2 | 验证 d3 |
| 3 / C+3 | d3 | H, d1, d2, d3 | 全接受后产生 bonus |

必须保留这个一位偏移：验证 d_i 使用输入 d_i **前一行**的 logits。把 d_i 自己所在行的 argmax 与 d_i 比较会得到错误验证。K 个候选需要 K 行验证分布，加上 1 行 bonus 分布；bonus 是从现有末行 logits 采样，不需要追加一次 target 前向。

以上 K+1 是正常解码入口的本请求查询长度。vLLM 同一批次可以包含长 prefill 或其他请求，本次 target 总输入行数不一定等于 `sum(K_i+1)`；每请求选择的采样行仍为 K_i+1。初版继续不混合 prefill 与验证，后续混合时必须分清前向行数和采样行数。

### 5.3 变长批次与两层行索引

参考 vLLM 使用扁平候选和每请求 K_i。令 S=`sum(K_i)`、B 为请求数，则选中 logits 形状为 `[S+B, vocab_size]`，候选为 `[S]`；target 验证行形状为 `[S, vocab_size]`，bonus 行为 `[B, vocab_size]`。

`logits_indices` 索引完整前向的 hidden states；`target_logits_indices` 和 `bonus_logits_indices` 则索引**已经选择并投影后的 logits**，不能混用两个坐标系。对请求 i，令 e_i 为其输入段在完整前向中的累计结束位置、s_i 为此前请求采样行数之和：

- `logits_indices` 选择 `[e_i-(K_i+1), ..., e_i-1]`。
- `target_logits_indices` 选择 `[s_i, ..., s_i+K_i-1]`。
- `bonus_logits_indices` 选择 `s_i+K_i`。
- `cu_num_draft_tokens` 为 K_i 的累计和；`cu_num_sampled_tokens` 为 K_i+1 的累计和。两者均为 B 项累计结束位置，不等同于通常包含起始 0 的 B+1 项 `cu_seqlens`。

例如 K_i=`[3,0,2]`，选中 logits 共 8 行，target 索引为 `[0,1,2,5,6]`，bonus 索引为 `[3,4,7]`，候选累计长度为 `[3,3,5]`，采样累计长度为 `[4,5,8]`。K_i=0 的请求没有验证行，只从其单行目标分布输出一个 token。vLLM 从选中的输入 token 中用 `target_logits_indices+1` 取出候选，显式落实预测行与输入位置的偏移。

### 5.4 贪心与随机接受逻辑

贪心模式对 K 行处理后的 target logits 求 argmax，按位置比较候选，只接受第一次不一致之前的连续前缀。vLLM 的贪心 Triton kernel 每请求遍历候选，在首个不一致位置写入该行 target argmax 作为 correction，后续槽位保留占位值；全接受才写入 bonus。批量前向与 GPU 行比较不改变“只能接受连续前缀”的逻辑。

例如候选 `[A,B,C]`，三行 target argmax 为 `[A,X,C]`，实际输出只能是 `[A,X]`。第三行即使匹配也依赖已拒绝的 B，不可采用；已有 bonus 分布同样失效，不能在 correction 后继续输出 bonus。全接受时输出 `[A,B,C,bonus]`。bonus 在 vLLM 中可以预先采样，但仅全接受时提交。

随机模式不能使用 argmax 相等判定。对候选 d_i 使用 `min(1,p_i(d_i)/q_i(d_i))` 接受概率，首个拒绝位置从归一化的 `max(p_i-q_i,0)` 取 correction；全接受从 p_(K+1) 取 bonus。n-gram 的 `draft_probs=None` 在 vLLM 中对应确定候选的点质量 q，候选处 q(d_i)=1，不代表跳过概率校正。目标 logits 的处理和采样约束必须符合实际配置，后续位置的处理应使用对应假设历史；本项目 P1 仍限定 temperature=0，随机模式留到 P4。

vLLM 输出缓冲区为 `[B,max(K_i)+1]`，未输出部分填占位 token。本项目可使用等价布局，但必须返回每请求有效长度，不能把占位值、拒绝尾部或未采用的 bonus 当作正式输出。

### 5.5 提交端点与当前实现的边界

若接受 a 个候选且无终止截断，输出为 `[d1, ..., da, correction_or_bonus]`，输出数 m=a+1。试算处理了 K+1 个输入，但正式只提交输入 `[u,d1,...,da]` 对应的 m 个位置：已计算长度为 **C+a+1=C+m**，正式历史长度为 **C+a+2=C+1+m**。最后输出的 correction/bonus 尚未作为 target 输入计算，留作下一轮锚点。首位拒绝时仍提交 u 的状态；全接受时提交全部 K+1 个输入状态。

EOS、`max_tokens` 或上下文边界可截断输出，必须按截断后的实际输出数重新确定状态端点，不能提交未输出尾部。若终止路径采用不同的最终锚点处理，应显式记录最终实际计算量，并单独验收。

单请求与变长批次的 `packed` 已采用一次 `[u,d1,...,dK]` 因果前向和全部 K+1 行投影；`sequential` 是逐词元正确性参考；`packed_guarded` 额外执行逐词元路径并比较预测/KV/GDN，属于本项目针对已发现数值分歧的保守措施。多请求的逐个 batch=1 参考已复现长输出分歧，因此批量 `sequential`/`packed_guarded` 当前使用普通 batch 形状的锚点参考，只提交一步目标输出，不计草稿接受；不同于单请求 K+1 行逐词元复核。上述 vLLM 正常验证链路没有逐候选重跑 target 来复核 packed logits。后续加速路径应以一次 packed 验证为目标；不能把 guarded 的复核成本计为 vLLM 式正常验证成本，也不能仅凭参考 vLLM 就移除现有正确性门槛。

## 6. GDN 与 KV 的事务式提交

### 6.1 首版正确性策略：私有状态试算与重放

不占用用于公共 prefix caching 的快照池。为每个活动验证请求准备私有临时 GDN slot，将当前卷积和 recurrent state 复制进去，用临时 slot 完成候选验证，正式 GDN 状态保持不变。

全部接受且无截断时，将临时最终状态复制回正式 slot。部分接受时，从保持不变的正式状态重放 `[u, d1, ..., da]`，仅更新正式状态，不进行第二次采样；主模型补偿词元仍保持未计算。发生截断时按截断后的端点提交/重放。

这个方案避免逆推 GDN，但状态复制和重放有明显开销。必须报告开销，不能因为减少验证轮数就宣称加速。草稿仍为 CPU n-gram。单请求保留同步参考入口；多请求已在 GPU 求接受端点和提交正式 token/长度，输出通过独立句柄异步拷回。显式 packed 的部分接受恢复需要取回长度来构造 GDN 重放输入，因此尚不是全程无同步的 GPU 流水线；批量保守模式只提交锚点参考的状态，无需部分接受重放。

### 6.2 KV 管理

按 C+K+1 的试算输入端点预留页；现有缓存页中已确认的位置不变，试算写入逻辑尾部。共享可写尾页需要检查并按需写时复制。拒绝尾部的物理 KV 可保留为不可见数据，但上下文长度必须立即恢复，后续写入覆盖；无继续使用价值的临时页应回收。

对试算区域禁止哈希发布和共享。只在正式提交后，按已确认且已计算的长度发布完整 KV 块；GDN prefix 快照必须对应同一端点。默认不增加解码阶段快照发布，在验证事务外保留当前预填充快照机制。

正式 GPU token、computed length、CPU Sequence、KV 页表及 GDN slot 的端点必须一致。事务完成前请求保持 `in_flight`，不能被再次调度、回收或复用；抢占和结束释放需等事务完成事件。

### 6.3 后续恢复优化

vLLM 的混合状态路径也需要按接受端点选择 recurrent state，不能仅回退 KV 长度。例如固定版本的 [`mamba_utils` 状态迁移](https://github.com/vllm-project/vllm/blob/a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb/vllm/v1/worker/mamba_utils.py#L519) 使用 `num_accepted_tokens-1` 的状态偏移；该字段的计数口径必须结合调用方核对，不能直接等同于本项目的草稿接受数 a。可借鉴“保留逐位置状态、按端点选择”的机制，但不能直接套用其 block 布局或索引。

对短候选链评估在 GDN 内核中保存逐位置状态，只恢复接受端点；或者保留分段检查点并有限重放。先核算每请求 `层数 × 头数 × V × K × 4 字节 × 保存端点数`，避免快照显存吞掉收益。不能用 prefix 快照池代替临时验证状态池。

## 7. 实施阶段与验收门槛

| 阶段 | 交付内容 | 完成标准 |
|---|---|---|
| P0：基础契约 | 配置、公共数据结构、长度/预测行定义、状态事务与确定性参考验证器 | mock logits 覆盖所有接受长度；默认关闭无行为变化 |
| P1：n-gram 贪心闭环 | CPU 草稿、单请求普通执行、私有 GDN 试算、部分接受重放、多词元输出 | 无匹配、全接受、首位/中间拒绝、EOS、长度截断均正确；不会发布拒绝历史 |
| P2：批量与系统集成 | 一次 target 因果前向、两层预测行索引、变长 K_i 元数据、GPU 接受/输出/提交、异步输出、资源不足回退 | K_i=0/1/4 混批的行映射与有效长度正确；slot 重排、跨页、并发、抢占重算、prefix 开关均通过状态对照 |
| P3：性能优化 | 逐位置状态/分段恢复、自适应 K、按接受收益回退、分段图验证 | 分项测量证明收益来源；低命中负载不持续承担无效验证成本 |
| P4：随机投机采样 | 草稿分布协议、拒绝采样、补偿分布、位置稳定随机计数 | 小词表统计验证主模型目标分布；拒绝后随机计数不受未提交候选污染 |
| P5：MTP | 目标隐藏状态导出、模型专属预测头与权重加载、草稿缓存同步 | 使用兼容且权重完整的模型验证；不能以空接口宣称支持 |
| P6：EAGLE/EAGLE3 | 对应训练草稿的特征层、投影、词表映射及缓存适配 | 先支持线性候选链，匹配 checkpoint 的特征契约；树验证另行设计 |
| P7：DFlash | 条件隐藏状态、块级并行草稿、草稿掩码与位置布局 | 草稿模型可用且兼容；草稿内部非自回归掩码不泄漏到主模型因果验证 |

P1 首先提供逐词元验证参考路径，再接入批量验证。参考路径不要求加速，是状态和预测行对齐的正确性基准。

P2 首版已按以下顺序接入，完整系统验收与 P3 优化继续按此顺序推进；实际测试结果以实施记录为准：

1. 已将第 5 节的行映射、首位/中间拒绝、全接受 bonus 和端点关系作为显式契约，核对当前 packed 实现；区分预测行错误与 packed/decode 数值差异。
2. P2 已建立变长 K_i 的打包输入和两层索引，在一次 target 前向中取得所有有效 logits；实现 GPU 最长接受前缀、有效输出长度及独立输出缓冲区。先验证 `[3,0,2]`、`[0,1,4]` 混批，覆盖拒绝后后续行恰好匹配但必须丢弃的情况。
3. 已接入真实输出长度驱动的 CPU/GPU 长度提交、拒绝尾部隐藏、GDN 恢复和异步完成事件；在正确性门槛通过后，P3 评估逐位置 GDN 状态替代部分接受重放，避免另一次采样或不必要的模型复核。

现有 prefill 扫描与 decode 递推可能存在 BF16 数值差异。P1/P2 要同时对比“无投机逐词元路径”和“相同分块方式的主模型参考”。已知 near-tie 单独记录；若批量验证导致无法满足已定义的精度/词元验收，不通过提高容差解决，使用保守回退或先修复差异。严格逐 token 一致不能仅从算法等价推出。

2026-09-28 扩展自然输入验收发现正常 packed 在中文、仓库实际代码和低匹配文本上出现真实词元分歧。因此同分块状态验收通过不能替代普通解码词元验收。开启后的默认模式已改为保守 `packed_guarded`；`sequential` 提供直接参考路径，`packed` 保留为显式实验选项。P2 的交付验收保留上述门槛，不能把这些分歧作为成功样例。

## 8. 随机采样扩展原则

常规线性投机采样在候选位置使用目标分布 p、草稿分布 q，接受概率为 `min(1, p(d)/q(d))`，拒绝后从归一化的 `max(p-q, 0)` 采样；全部接受时从目标末行分布采样额外词元。分布必须包含实际温度及届时已支持的采样处理。

n-gram 的确定候选可视为点质量 q，而不是“没有草稿概率”。MTP/EAGLE/DFlash 后端必须提供与实际候选生成相符的分布契约；并行草稿是否适用同一线性拒绝公式需依据其联合/条件分布核实，不能机械套用。

草稿、接受判定和目标采样使用独立的随机计数域，基于请求与逻辑位置定义。随机结果要求分布正确；不承诺与非投机路径同 seed 的样本逐词元相同。贪心与随机验收分别报告。

## 9. MTP、EAGLE 与 DFlash 的扩展边界

MTP 需要匹配目标架构的预测头及训练权重。当前加载器跳过 MTP 参数，必须显式注册并验证加载完整性；本地 Qwen3.5-0.8B 是否具备可用预测头需检查 checkpoint，不能预设存在。

EAGLE 家族需要兼容目标模型的训练草稿，使用的目标特征层、归一化位置和词表映射可能不同。目标模型提供可选的层级特征导出接口，按实际需要分配；没有模型草稿时，不额外保存全部隐藏层。

DFlash 使用目标隐藏状态条件下的块级并行草稿模型，草稿网络的计算和注意力布局与自回归主模型不同。它可以输出统一候选链供验证，但其模型执行不能伪装成 n-gram 或普通自回归循环。

未来每种方法的“支持”均需包括真实模型加载、草稿执行、拒绝恢复、验收脚本和性能记录。准备好接口只是架构准备，不是完成模型支持。

## 10. 验证与测量计划

已提供 `tests/test_spec_decode.py`、`tests/test_spec_state.py`、`tests/test_spec_execution.py`、`benchmarks/validate_spec_decode.py` 和 `benchmarks/bench_spec_decode.py`。扩展自然输入与 near-tie 诊断使用 `benchmarks/validate_spec_natural.py`，诊断回归使用 `tests/test_spec_diagnostics.py`。

预测行专项验收：使用各行不同的 mock logits 检查一位偏移；检验完整 hidden-state 索引与选中 logits 索引两个坐标系；变长 K_i=0/1/4 混批不得串请求；对每个接受长度 a 验证输出数 a+1、bonus 条件、占位过滤及 C+a+1 的状态端点。真实模型检查一次 packed 前向确实投影全部有效行，分别统计正常验证、guarded 复核和部分接受状态重放的前向次数。

新增 `tests/test_spec_batch.py` 与 `benchmarks/validate_spec_batch.py` 检查变长批次、CUDA 接受/提交、独立输出句柄及真实模型恢复端点。单请求入口仍保留原诊断钩子，批量路径通过 `verify_speculative_batch` 独立验收。

测试覆盖：n-gram 重叠/多匹配/短续段、候选全拒绝至全接受、预测行偏移、EOS 在候选和补偿词元内、剩余输出长度、上下文末尾、页边界前后、共享 prefix、状态池淘汰、资源不足、取消/抢占、异步输出所有权、非活动 slots 不变及异常事务清理。

完整对照包括已确认 token、KV 有效区域、conv/recurrent state、CPU/GPU 已计算长度、页引用计数；禁止只检查最终文本。prefix 对照须记录双方实际命中与冷/热条件：普通解码自身的冷/热数值差异单独报告，相同缓存条件下的词元与状态仍要求精确一致，不通过扩大容差豁免。小词表随机采样做概率统计验证，真实模型另做数值与生成验收。

性能固定模型、硬件、精度、输入和输出长度及采样策略，对照当前非投机版本。场景包括重复文本/代码、普通对话、低匹配历史；batch=1/4/8/16，K=1/2/4/8，分别测量纯解码和后续集成的混合到达场景。

记录输出 token 吞吐、整段生成耗时、逐轮耗时、草稿匹配率、候选接受率、每次验证输出数、草稿/验证/状态复制/重放/提交时间、峰值显存和回退次数。没有流式服务接口时不把离线指标标为服务 TTFT/P95 TPOT。预热与重复次数、模型/框架版本均写入 JSON；外部 vLLM 对照单独运行同条件配置，不预填性能目标或加速数字。

## 11. 官方参考

以下资料用于核对算法接口与扩展方向，本文方案基于 HybridInfer 的现有混合状态实现，不照搬 vLLM 的内部缓存布局。

- [vLLM RejectionSampler 官方 API 与源码说明](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/rejection_sampler/)（第 5 节实现结论以固定提交为准）
- [vLLM 投机解码概览](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
- [n-gram 草稿](https://docs.vllm.ai/en/stable/features/speculative_decoding/n_gram/)
- [MTP](https://docs.vllm.ai/en/stable/features/speculative_decoding/mtp/)
- [EAGLE 草稿](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/)
- [投机配置与方法能力](https://docs.vllm.ai/en/stable/api/vllm/config/speculative/)
- [DFlash 方法说明](https://docs.vllm.ai/projects/speculators/en/v0.7.0/user_guide/algorithms/dflash/)

P0–P1 与 P2 首版执行链路已接入；后续补齐 P2 系统验收，再依据正确性和状态恢复成本推进 P3；MTP/EAGLE/DFlash 保留公共接口与明确依赖，不在首轮创建未验证的模型实现。
