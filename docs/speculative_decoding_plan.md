# 投机解码实施计划

更新：2026-09-29。目标是数学语义一致、正确选择原 trial 端点、控制数值误差，
并通过实测建立性能收益。跨浮点路径的逐位或 greedy 全序列一致性作为诊断指标。
预测行、端点、历史范围、EOS/长度与采样算法仍严格验收。

当前实现见 [进度](speculative_decoding_progress.md) 和 [公共链路与 MTP](speculative_mtp_implementation.md)，
结果见 [MTP 评估](mtp1_evaluation.md)，脚本见 [验收指南](../benchmarks/README.md)。

## 1. 当前范围

已实现 Qwen3.5 Dense、单 GPU、线性候选、变长批量 target 验证、共享随机拒绝采样
和真实 checkpoint MTP。n-gram 用于公共链路测试。投机默认关闭，开启后默认
`packed_guarded`；显式 `packed` 执行多词元接受与原 trial 端点提交。

MTP 当前要求 eager、关闭 prefix cache，使用单个预测层、共享 embedding/head、
独立 draft KV 和 target feature 历史。草稿为 greedy 点质量，可服务随机 target。
当前不混合 prefill 与投机验证。树候选需独立 mask、位置和 GDN 分支状态。
最终草稿范围为 MTP、EAGLE-3、P-EAGLE、DFlash、DFlash2、DSpark；后五种尚未实现。

## 2. 公共架构

| 模块 | 当前职责 |
|---|---|
| `config.py`、`interfaces.py` | 方法、候选上限、预算、CPU/设备草稿与后端能力 |
| `metadata.py` | 变长累计长度、hidden/logits 两套行索引 |
| `batch_execution.py` | 一次 target 前向、试算与原 trial 端点提交 |
| `batch_verifier.py`、`rejection.py` | GPU 接受、补偿/bonus 与停止条件 |
| `state.py`、`endpoints.py` | 状态事务、共享 conv 历史与 recurrent 端点 |
| `commit.py`、`async_output.py` | 有效历史/长度提交、输出缓冲区及完成事件 |
| `mtp.py`、`models/qwen3_5_mtp.py` | 特征历史、draft KV、真实模型与权重 |
| `ngram.py`、`verifier.py` | 历史匹配草稿和 CPU 接受参考 |

设备候选在调度适配边界仍转为 CPU tuple，当前不是全设备调度。
所有请求共用 `verify_speculative_batch`；指标由 runner/scheduler 记录。

## 3. Target 验证与接受

### 3.1 一次因果前向，K+1 行预测

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

以上 K+1 是正常解码入口的本请求查询长度。vLLM 同一批次可以包含长 prefill 或其他请求，本次 target 总输入行数不一定等于 `sum(K_i+1)`；每请求选择的采样行仍为 K_i+1。当前不混合 prefill 与验证，后续混合时必须分清前向行数和采样行数。

### 3.2 变长批次与两层行索引

参考 vLLM 使用扁平候选和每请求 K_i。令 S=`sum(K_i)`、B 为请求数，则选中 logits 形状为 `[S+B, vocab_size]`，候选为 `[S]`；target 验证行形状为 `[S, vocab_size]`，bonus 行为 `[B, vocab_size]`。

`logits_indices` 索引完整前向的 hidden states；`target_logits_indices` 和 `bonus_logits_indices` 则索引**已经选择并投影后的 logits**，不能混用两个坐标系。对请求 i，令 e_i 为其输入段在完整前向中的累计结束位置、s_i 为此前请求采样行数之和：

- `logits_indices` 选择 `[e_i-(K_i+1), ..., e_i-1]`。
- `target_logits_indices` 选择 `[s_i, ..., s_i+K_i-1]`。
- `bonus_logits_indices` 选择 `s_i+K_i`。
- `cu_num_draft_tokens` 为 K_i 的累计和；`cu_num_sampled_tokens` 为 K_i+1 的累计和。两者均为 B 项累计结束位置，不等同于通常包含起始 0 的 B+1 项 `cu_seqlens`。

例如 K_i=`[3,0,2]`，选中 logits 共 8 行，target 索引为 `[0,1,2,5,6]`，bonus 索引为 `[3,4,7]`，候选累计长度为 `[3,3,5]`，采样累计长度为 `[4,5,8]`。K_i=0 的请求没有验证行，只从其单行目标分布输出一个 token。vLLM 从选中的输入 token 中用 `target_logits_indices+1` 取出候选，显式落实预测行与输入位置的偏移。

### 3.3 贪心与随机接受逻辑

贪心模式对 K 行处理后的 target logits 求 argmax，按位置比较候选，只接受第一次不一致之前的连续前缀。vLLM 的贪心 Triton kernel 每请求遍历候选，在首个不一致位置写入该行 target argmax 作为 correction，后续槽位保留占位值；全接受才写入 bonus。批量前向与 GPU 行比较不改变“只能接受连续前缀”的逻辑。

例如候选 `[A,B,C]`，三行 target argmax 为 `[A,X,C]`，实际输出只能是 `[A,X]`。第三行即使匹配也依赖已拒绝的 B，不可采用；已有 bonus 分布同样失效，不能在 correction 后继续输出 bonus。全接受时输出 `[A,B,C,bonus]`。bonus 在 vLLM 中可以预先采样，但仅全接受时提交。

随机模式不能使用 argmax 相等判定。对候选 d_i 使用 `min(1,p_i(d_i)/q_i(d_i))` 接受概率，首个拒绝位置从归一化的 `max(p_i-q_i,0)` 取 correction；全接受从 p_(K+1) 取 bonus。n-gram 的 `draft_probs=None` 在 vLLM 中对应确定候选的点质量 q，候选处 q(d_i)=1，不代表跳过概率校正。目标 logits 的处理和采样约束必须符合实际配置，后续位置的处理应使用对应假设历史；共享随机拒绝采样已实现；当前 MTP 使用 greedy 点质量草稿。

vLLM 输出缓冲区为 `[B,max(K_i)+1]`，未输出部分填占位 token。本项目可使用等价布局，但必须返回每请求有效长度，不能把占位值、拒绝尾部或未采用的 bonus 当作正式输出。

### 3.4 提交端点与当前实现的边界

若接受 a 个候选且无终止截断，输出为 `[d1, ..., da, correction_or_bonus]`，输出数 m=a+1。试算处理了 K+1 个输入，但正式只提交输入 `[u,d1,...,da]` 对应的 m 个位置：已计算长度为 **C+a+1=C+m**，正式历史长度为 **C+a+2=C+1+m**。最后输出的 correction/bonus 尚未作为 target 输入计算，留作下一轮锚点。首位拒绝时仍提交 u 的状态；全接受时提交全部 K+1 个输入状态。

EOS、`max_tokens` 或上下文边界可截断输出，必须按截断后的实际输出数重新确定状态端点，不能提交未输出尾部。若终止路径采用不同的最终锚点处理，应显式记录最终实际计算量，并单独验收。

所有请求（包括 B=1）统一通过 `verify_speculative_batch` 执行。Native `packed` 对
`[u,d1,...,dK]` 做一次因果前向并投影全部有效预测行，提交原 trial 的接受端点。
`sequential` 直接执行普通 batch 形状的单步 anchor；`packed_guarded` 先做 packed
诊断，再提交普通单步 anchor，接受候选数为零。两者均为参考/回退路径，不将
其额外开销计入正常 native 验证成本。

## 4. GDN 与 KV 的事务式提交

### 4.1 当前策略：私有状态试算与原 trial 端点提交

验证开始时，每个请求的正式 conv/recurrent 状态复制到私有 trial slot；保存原状态
用于异常恢复。Packed scan 保存每个 recurrent 端点，conv 保存初始历史和原始
trial token 的共享扩展历史。正式状态在接受判定前不推进。

接受或截断后，GPU 按实际输出长度选择原 trial 端点，提交 conv 窗口和 recurrent
状态，再提交正式 token 与 computed length。Correction/bonus 保持未计算，作为
下轮 anchor。Native 不重跑 target；端点缺失视为实现错误，恢复事务并报错。
显式资源不足回退则重置私有状态，执行普通单步 anchor，并复制该端点回正式 slot。

事务接口区分 `commit_trial()`（普通 anchor 状态复制）与
`finish_endpoint_commit()`（端点选择完成）。异常仍恢复原正式状态。状态复制、
选择及提交成本分别测量，不能仅凭减少验证轮数宣称加速。

### 4.2 KV 管理

按 C+K+1 的试算输入端点预留页；现有缓存页中已确认的位置不变，试算写入逻辑尾部。当前共享可写尾页回退普通解码；后续写时复制须独立验收。拒绝尾部的物理 KV 可保留为不可见数据，但上下文长度必须立即恢复，后续写入覆盖；无继续使用价值的临时页应回收。

对试算区域禁止哈希发布和共享。只在正式提交后，按已确认且已计算的长度发布完整 KV 块；GDN prefix 快照必须对应同一端点。默认不增加解码阶段快照发布，在验证事务外保留当前预填充快照机制。

正式 GPU token、computed length、CPU Sequence、KV 页表及 GDN slot 的端点必须一致。事务完成前请求保持 `in_flight`，不能被再次调度、回收或复用；抢占和结束释放需等事务完成事件。

### 4.3 状态预算

Conv 保存初始历史加 trial raw tokens，按端点选择历史窗口；recurrent 保存各输入端点。
默认状态快照预算 256 MiB，调度按实际布局限制验证输入，K 是上限，实际 K_i 可不同。
本模型 18 层 FP32 recurrent 每请求每端点约 18 MiB，B=8、K=4 的端点约 720 MiB。
不能假定大批次均有完整候选数；prefix 快照池与试算状态池独立。
优化存储与选择成本时保持原 trial 端点契约，native 部分接受不重跑 target。

## 5. 下一阶段建议：MTP 性能与覆盖

以下任务尚未实施，按测量结果调整优化优先级：

1. **完整候选扫描与耗时分解。** 五类自然输入，K=1/2/4、B1/B4，固定模型、
   精度、输入、输出预算、采样与缓存条件；配对轮换并增加重复次数。
   记录实际 K_i、每轮有效输出、接受率、proposer/target/状态提交时间、CPU 同步、
   峰值显存与回退。当前性能脚本只支持 B1，B4 需扩展入口。
2. **普通图路径对照。** 当前 MTP 强制 eager，需扩展测试入口，独立配置普通
   decode graph baseline 与 MTP eager。保留 eager/eager 对照以定位成本，
   实际收益以当前最快普通路径为参照。
3. **优化主要开销。** 依据 profiling 实现按候选步合批的 MTP proposer，复用
   元数据/缓冲区、减少回传与同步，再推进固定形状 proposer/target CUDA Graph。
   变长 padding 不得推进正式历史或状态。
4. **候选长度选择和低收益回退。** 根据有效输出与每轮实际成本选择 K，接受率
   仅为输入指标；覆盖低匹配、请求结束、资源压力及混合 K_i。
5. **扩大质量与系统覆盖。** 覆盖 K=2/4、生产 packed LM head、更多客观任务、
   自然生成退化、随机 target、动态到达、取消/抢占与容量压力。每次优化伴随
   原 trial 端点、draft-cache 生命周期与数值/质量检查。

交付需有可复现结果、收益范围及无收益场景，不预设加速倍数。随后推进其他真实草稿后端。

## 6. 后端扩展契约

固定参考路由与独立契约见 [后端适配](vllm_speculative_alignment.md)。
EAGLE-3/P-EAGLE 需要 checkpoint 对应的层特征、词表映射、查询布局与缓存；
P-EAGLE 需要并行训练权重。DFlash 家族需要独立 context/query KV、mask 与位置。
DFlash2 selector、DSpark Markov head 的实际条件分布必须用于 q。
每个后端均需真实权重、执行、恢复、验收与性能产物。

## 7. 验收标准

- **算法与系统：** 同 logits/概率的接受、补偿与 bonus 语义正确；预测行、
  EOS/长度、历史/页引用、事务清理精确正确。K_i=0/1/4 混批不串请求。
- **状态选择：** oracle 来自同一次原 trial；实际输出 m 个 token 时选零基端点
  m-1，computed 推进 C+m，最后输出保持未计算；拒绝尾部不进入后续 draft 条件。
- **数值：** 同输入同初始状态做 reset/rolling TV/KL、argmax 翻转、margin 和
  层/状态偏差检查，运行前声明预算。当前诊断两边 LM head 逐行投影，生产 packed
  head 另由自由生成与质量检查覆盖。未声明预算只报告诊断。
- **质量：** 固定任务及评分，对照正确率和异常/重复/截断等退化；12 道算术题
  仅为 smoke test，不能证明通用质量无损。
- **性能：** 无 intrusive 状态探针，baseline 关闭 MTP feature 记录；报告配对
  时间比、吞吐与整段耗时。不同输出的耗时比仅表示实际路径测量，不是同轨迹加速。
  离线结果不标为服务 TTFT/P95 TPOT。

固定 vLLM 源码差分与安装版完整引擎对照分别声明版本及范围；token 相同率不替代
算法、端点、数值预算或质量验收。入口与命令见 [验收指南](../benchmarks/README.md)。
