# vLLM 参考与模型草稿适配契约

更新：2026-09-30。当前公共链路、真实 MTP 与 Qwen3.5-2B DFlash/DSpark 已实现；各草稿适配共用 target 验证和状态提交，
各草稿模型的特征、查询布局、缓存与实际 q 分别实现。

## 参考范围

源码差分固定提交 `a4eb3f25d6f9b3cad7ecf5390423d853935fcaeb`，本地 checkout 为
`/home/lang/workspace/vllm`。脚本核对 revision、源文件与 SHA256 并执行原定义。
完整模型参考是安装版 vLLM 0.19.0，不能与固定提交源码差分混为同一版本。

| 检查 | 已验证范围 |
|---|---|
| `validate_vllm_spec_alignment.py` | 元数据 CPU/GPU 各 104 布局；新旧 greedy kernel 112 批/931 请求 |
| `validate_vllm_random_rejection.py` | 同 p/q、uniform、recovered/bonus 输入，200 批/1000 请求 |
| `validate_vllm_gdn_recurrent.py` | recurrent 72 布局/1386 端点 |
| `validate_vllm_spec_conv.py` | BF16/FP16 conv 8 布局/140 端点 |
| `validate_target_norm.py` | 安装版原 static norm，固定编译配置及操作数布局 |

这些是局部算法/kernel 契约，不能证明整模型跨引擎数值或生成一致。
随机数生成器不同，不保证同 seed token 相同；同概率接受与补偿语义必须正确。
完整模型对照记录 token 差异，数值与质量预算独立验收。入口见 [验收指南](../benchmarks/README.md)。

## 当前实现

Native target 使用多词元 recurrent 验证，保存原 trial 端点，部分接受直接选择状态，
保留有效 KV；不通过重跑较短 target 恢复。Conv 使用共享扩展历史。
设备草稿与按需目标特征接口、共享随机拒绝 sampler、独立 MTP draft KV 均已接入。
调度边界仍将设备候选转 CPU tuple。

本地 Qwen3.5-0.8B 有一个 MTP 层，15 个 checkpoint 张量已完整加载；合并 MLP
分片后 14 个参数，20,452,864 个元素，BF16 约 39.01 MiB，不含共享 embedding/head
与 draft KV。执行顺序及缓存同步见 [MTP 实现](speculative_mtp_implementation.md)。

## 六种后端契约

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

## 公共状态与特征约束

- 目标层特征采集点、层序、residual/归一化与 token shift 必须匹配 checkpoint。
- 概率草稿提供实际条件 q；DFlash2 selector 与 DSpark Markov 修正不能省略。
- Target 输入 `[u,d1,...,dK]` 与 draft 的 K/K+1 query 布局分开定义。
  输出 m 个 token 时选原 trial 零基端点 m-1，computed 推进 C+m，最后输出未计算。
- 接受后的特征/draft KV 仅依赖已确认历史；结束、抢占与异常不留下未来候选条件。
- Recurrent 端点按实际预算分配。本模型每请求每端点约 18 MiB；B=8、K=4 约
  720 MiB，不能不计显存地扩大候选与批次。

## 推进顺序

先完成 [MTP 性能与覆盖计划](speculative_decoding_plan.md)，
EAGLE-3 已完成线性协议适配，仍需配套训练权重；DFlash/DSpark 已接入真实 Qwen3.5-2B checkpoint，结果见 [块草稿说明](block_draft_implementation.md)。P-EAGLE 和 DFlash2 仍待实现。当前配置支持 `ngram`、`mtp`、`eagle3`、`dflash`、`dspark`。
