## MTGR: Industrial-Scale Generative Recommendation Framework in Meituan

### 一句话定位

MTGR 是美团提出的工业级生成式推荐排序框架：用 HSTU-style self-attention 把“用户、多段行为序列、实时行为、多个候选及交叉特征”统一成 token 序列，在保留传统 DLRM 强 cross features 的同时，实现可扩展的大模型化推荐排序。

### 基本信息

- **论文**：MTGR: Industrial-Scale Generative Recommendation Framework in Meituan
- **arXiv**：2505.18654
- **作者**：Ruidong Han, Bin Yin, Shangyu Chen 等，美团
- **发布时间**：2025-05-24
- **场景**：美团外卖主排序流量
- **核心关键词**：
  - Generative Recommendation
  - DLRM
  - HSTU
  - Cross Features
  - User-level Compression
  - Group-Layer Normalization, GLN
  - Dynamic Masking
  - Scaling Law

> 注：本文依据 arXiv TeX 源码、摘要、正文片段和 3 张图进行整理；在线实验部分在给定 excerpt 中不完整，因此在线收益只记录论文摘要和可见内容，不展开具体数值。

### 摘要中文翻译

Scaling law 已在 NLP 和 CV 中被广泛验证。推荐系统中，近期工作开始采用生成式推荐来提升可扩展性，但这些方法通常需要放弃传统推荐模型中精心构造的 cross features。作者发现，这会显著损害模型效果，而且单纯扩大模型规模无法弥补。

为解决这个问题，论文提出 **MTGR, Meituan Generative Recommendation**。MTGR 基于 **HSTU** 架构建模，同时保留原始 DLRM 特征，包括 cross features。为保证高效扩展，MTGR 通过 **user-level compression** 同时加速训练和推理。论文还提出 **Group-Layer Normalization, GLN**，用于提升不同语义空间下的编码效果；提出 **dynamic masking**，避免信息泄漏。

此外，作者优化了训练框架，使模型可以支持相对 DLRM 高 10 到 100 倍计算复杂度的模型，而训练成本没有显著增加。MTGR 的单样本前向推理 FLOPs 达到 DLRM 的 **65 倍**，并取得近两年来美团离线和在线最大收益，已部署在美团外卖主流量。

### 研究问题

#### 1. 推荐系统里的 scaling dilemma

传统排序模型通常是 DLRM-like 流程：

1. 用户特征、历史序列、实时行为、候选 item、cross features 分别 embedding；
2. 用 target attention 建模候选 item 与用户历史行为；
3. 拼接所有特征；
4. MLP 输出每个候选的 logit。

对第 \(i\) 个候选，传统输入可写成：

\[
D_i = [U, S, R, C_i, I_i]
\]

其中：

- \(U\)：用户 profile，如年龄、性别；
- \(S\)：长期历史行为序列；
- \(R\)：实时行为序列，real-time actions；
- \(C_i\)：用户-候选相关的 cross features；
- \(I_i\)：候选 item 特征。

传统架构的问题是：

| 扩展方向 | 好处 | 问题 |
|---|---|---|
| 扩大 user module | 用户表示更强，可被多个候选复用 | 用户-候选交互不足 |
| 扩大 cross module / MLP | 用户-候选交互更强 | 对每个候选都要算，推理成本随候选数线性增长 |

因此推荐排序的大模型化面临矛盾：

> 想要强交互，就要对每个 candidate 计算；想要高效扩展，就必须复用用户侧计算。

#### 2. 生成式推荐的问题：丢掉 cross features

很多 generative recommendation 方法把推荐建模为 item/token 生成或序列预测，从而更容易享受 scaling law。

但工业推荐系统里，cross features 非常关键，例如：

- 用户对某 item 的历史 CTR；
- 用户和某品类的交互统计；
- item 与地理、时间、场景的交叉统计；
- 用户-item、用户-category、item-spatio-temporal 特征。

论文发现：

> 如果为了生成式框架放弃这些 cross features，效果会显著下降，而且扩大模型规模也补不回来。

这就是 MTGR 要解决的基础问题：

> 如何让推荐排序像大模型一样可扩展，同时不丢掉工业 DLRM 中最有效的 cross features？

### 核心方法

### 1. User Sample Aggregation：把多个候选合成一个用户级样本

传统排序是每个候选一个样本：

\[
D_i = [U, S, R, C_i, I_i]
\]

MTGR 改成按用户聚合 \(K\) 个候选：

\[
D = [U, S, R, [C, I]_1, \ldots, [C, I]_K]
\]

核心变化：

- 同一个 request 或训练窗口内，同一用户的多个候选被放进一个样本；
- 用户特征 \(U\)、历史序列 \(S\)、实时行为 \(R\) 只算一次；
- 每个候选 \([C_i, I_i]\) 作为一个 candidate token；
- 输出时一次前向得到所有候选的 logits。

这相当于做了 **user-level compression**：

> 计算粒度从“用户-候选 pair”压缩到“用户 request / 用户窗口”。

推理成本因此不再严格随候选数线性增长，为扩大模型提供空间。

### 2. Token 化：把推荐特征统一成序列

MTGR 把不同来源的推荐特征统一成 token：

- 用户 scalar features：每个 scalar feature 变成一个 token；
- 历史行为序列 \(S\)：每个 item 变成一个 token；
- 实时行为序列 \(R\)：每个实时交互 item 变成一个 token；
- 候选 item + cross features：每个候选变成一个 token。

候选 token 由 item features 和 cross features 拼接后经 MLP 得到：

\[
E_{cand,i} = \mathrm{MLP}(\mathrm{Concat}(I_i, C_i))
\]

最终输入 token 序列：

\[
X_D = \mathrm{Concat}([E_U, E_S, E_R, E_C])
\]

其中 \(E_C\) 表示所有 candidate tokens。

### 3. Unified HSTU Encoder：用 self-attention 建模所有特征交互

MTGR 使用 HSTU-inspired encoder-only 架构。

每层大致流程：

1. 输入 token 序列 \(X\)；
2. 先做 Group LayerNorm；
3. 投影得到 \(Q, K, V, U\)；
4. 计算带 SiLU 的 attention；
5. 加 customized mask；
6. value update；
7. 与 \(U\) 做 element-wise / dot-product interaction；
8. 再做 Group LayerNorm 和残差；
9. 堆叠 \(L\) 层。

论文片段中的 attention 形式可概括为：

\[
A = \mathrm{silu}(QK^\top) \odot M
\]

\[
V' = A V
\]

然后与 \(U\) 交互并进入后续 MLP / residual。

与传统 DLRM 的差异：

- DLRM：特征交互主要靠 target attention + MLP；
- MTGR：所有用户、序列、实时行为、候选、cross features 都进入 unified self-attention；
- 候选之间也在同一序列中，但通过 mask 控制不能泄漏信息。

### 4. Group-Layer Normalization, GLN

不同推荐特征来自不同语义空间：

- 用户 profile；
- 长期行为序列；
- 实时行为序列；
- 候选 item；
- cross features。

如果直接拼成一个 token 序列做 LayerNorm，分布差异可能影响 attention。

MTGR 提出 **Group-Layer Normalization, GLN**：

> 按特征语义域分组做归一化，使不同 domain 的 token 在进入 self-attention 前分布更对齐。

直观理解：

- user tokens 和 candidate tokens 不是同一种语义；
- sequence item tokens 和 cross-feature tokens 统计分布也不同；
- GLN 是推荐特征版的 domain-aware normalization。

消融显示，去掉 GLN 会明显掉点。

### 5. Dynamic Masking：避免实时行为信息泄漏

MTGR 的输入中有 \(R\)：用户实时行为序列。

问题是：

- 一个训练聚合窗口里可能包含多个时间点的候选；
- 某些实时行为发生在某个候选曝光之后；
- 如果该候选能 attend 到未来实时行为，就会出现 label leakage / information leakage。

因此 MTGR 设计 customized dynamic mask：

规则：

1. **Static sequence 全可见**  
   用户 profile \(U\) 和历史序列 \(S\) 被视为静态信息，对所有 token 可见。

2. **Real-time sequence 按时间因果可见**  
   实时行为 \(R\) 按时间顺序，只允许候选看到其发生之前的信息。

3. **Candidate token 只看自己**  
   候选之间不能互相泄漏信息；candidate token 对其他 candidate 不可见。

这不是标准 causal mask，而是推荐场景下的时间感知 mask。

### 6. 训练系统优化

MTGR 相比 DLRM 计算复杂度显著增加，因此训练系统必须重构。

作者从 TensorFlow 转到 PyTorch / TorchRec 生态，并做了多项优化：

#### Dynamic Hash Table

TorchRec 默认 fixed-size embedding table 不适合工业流式训练：

- 新用户、新 item 不断出现；
- 静态表容量容易溢出；
- 预留太大又浪费显存 / 内存。

MTGR 使用动态哈希 embedding table：

- key storage 和 value storage 解耦；
- key 只保存 ID 到 embedding 指针的映射；
- value 保存 embedding vector 和 metadata；
- 支持动态扩容和 eviction。

#### Embedding Lookup 优化

跨设备 embedding lookup 需要 all-to-all 通信。

作者用两阶段去重：

- 通信前去重；
- 通信后再保证 ID 唯一；

减少重复 ID 传输。

#### Load Balance：动态 batch size

用户行为序列长度长尾严重。

固定 batch size 会导致：

- 某些 GPU 分到长序列，计算很慢；
- 其他 GPU 等待。

MTGR 使用 dynamic batch size：

- 每张 GPU 根据实际序列长度调整 local batch size；
- 保证计算负载接近；
- 梯度聚合时按 batch size 加权，保持与固定 batch 逻辑一致。

#### Pipeline 与 kernel 优化

使用三条 stream：

- copy stream：CPU 到 GPU 数据拷贝；
- dispatch stream：embedding lookup 和通信；
- compute stream：forward / backward。

同时使用：

- bf16 mixed precision；
- 基于 cutlass 的 specialized attention kernel。

训练效果：

- 相比 TorchRec，吞吐提升 **1.6x–2.4x**；
- 支持 100+ GPUs；
- 相比 DLRM，单样本前向 FLOPs 可达 **65x**，但训练成本接近不变。

### 关键图表解读

#### 图 1：MTGR 整体结构

图中上半部分展示了 MTGR 的数据重排与架构：

- Raw Features 包括：
  - User：age、gender；
  - Seq：长期历史 item；
  - RT：实时行为 item；
  - Candidates：每个候选包含 ctr、pv、ID、tag、brand。
- 经过 embedding lookup 后：
  - scalar user feature 直接变 token；
  - Seq 和 RT 经各自 MLP 变 token；
  - Candidate 的 item feature 与 cross feature 一起经 MLP 变 candidate token。
- 所有 token 拼接后进入多层 self-attention。
- 最后只取 candidate token 的输出，经 MLP 得到每个候选 logit。

关键点：

> MTGR 没有放弃 cross features，而是把 cross features 并入 candidate token，让它参与全局 self-attention。

#### 图 1(b)：Self-Attention Block

结构是 HSTU-like：

- 输入 token 先经过 Group LN；
- 分别投影到 \(Q, K, V, U\)；
- \(QK^\top\) 计算 attention；
- 加 mask；
- 更新 value；
- 与 \(U\) 交互；
- 再经过 Group LN 和残差连接。

它不是标准 Transformer block，而是面向推荐序列建模优化过的 HSTU-style block。

#### 图 1(c)：Customized Mask

mask 矩阵里：

- age、ctr 等静态 token 对所有 token 可见；
- seq1、seq2 作为历史序列，对后续建模可见；
- rt1、rt2 按时间因果可见；
- target1、target2、target3 只能看自己和合法历史信息；
- target 之间不互相可见。

这解决了推荐训练中非常实际的问题：

> 聚合多个候选和实时行为后，如果不做时间 mask，模型会偷看未来行为。

#### 图 2：传统 DLRM 流程

传统系统是 candidate-wise ranking：

- 每个候选单独处理；
- candidate item 作为 query 去 attend 用户历史和实时行为；
- user、seq、rt、cross、candidate 拼接；
- MLP 输出单个 logit。

问题：

> 候选数越多，cross module 重复计算越多，扩大模型会直接带来推理延迟爆炸。

MTGR 的 user-level aggregation 正是为了打破这一点。

#### 图 3：Scaling Law 结果

图中展示了 MTGR 对三个维度的扩展：

1. HSTU block 数量增加：
   - 2 层：CTCVR GAUC 0.6591
   - 3 层：0.6603
   - 5 层：0.6615
   - 8 层：0.6623

2. \(d_{model}\) 增大：
   - 128：0.6543
   - 256：0.6583
   - 512：0.6603
   - 1024：0.6614

3. sequence length 增大：
   - 100：0.6551
   - 300：0.6591
   - 1000：0.6603
   - 5000：0.6625

4. FLOPs 与收益呈近似 power-law：
   - 横轴是 \(\log_2(\mathrm{FLOPs})\)；
   - 纵轴是相对 UserTower-SIM 的 CTCVR GAUC gain；
   - 点大致落在一条上升直线上。

结论：

> MTGR 的收益确实随模型深度、宽度、序列长度和计算量平滑提升，表现出推荐场景中的 scaling behavior。

### 关键贡献

1. **指出工业生成式推荐的关键缺陷**  
   现有 generative recommendation 往往丢弃 DLRM cross features，而 cross features 在工业推荐中极其重要。

2. **提出保留 cross features 的 generative ranking framework**  
   MTGR 把用户、序列、实时行为、候选和 cross features 统一 token 化，并用 HSTU-style self-attention 建模。

3. **User-level compression 降低推理复杂度**  
   多个候选聚合为一个用户级样本，一次前向输出多个 logits，为模型 scaling 留出计算预算。

4. **GLN 解决多语义空间 token 分布不一致问题**  
   对不同 feature group 做归一化，提升 unified encoder 的建模稳定性。

5. **Dynamic masking 避免信息泄漏**  
   针对实时行为和候选曝光时间设计因果可见性，而不是简单套 causal mask。

6. **完成工业级训练系统优化**  
   动态哈希表、embedding lookup 去重、动态 batch size、pipeline、多流、bf16、自定义 attention kernel，使 65x FLOPs 模型可训练、可上线。

### 实验与结论

#### 数据集

使用美团工业日志，而不是公开数据集。

原因：

- 公开数据集很少包含丰富 cross features；
- 工业推荐的 cross features 是核心能力；
- 美团数据规模足够大，可以验证复杂模型是否能充分收敛。

10 天离线数据规模：

| Split | Users | Items | Exposure | Click | Purchases |
|---|---:|---:|---:|---:|---:|
| Train | 0.21B | 4,302,391 | 23.74B | 1.08B | 0.18B |
| Test | 3,021,198 | 3,141,997 | 76,855,608 | 4,545,386 | 769,534 |

任务：

- CTR；
- CTCVR。

指标：

- AUC；
- GAUC，按用户分组的 AUC，更关注同一用户内候选排序能力。

#### 模型规模

| Model | Setting | GFLOPs / example |
|---|---|---:|
| UserTower-SIM | baseline | 0.86 |
| MTGR-small | \(n_{layer}=3, d_{model}=512, n_{heads}=2\) | 5.47 |
| MTGR-medium | \(n_{layer}=5, d_{model}=768, n_{heads}=3\) | 18.59 |
| MTGR-large | \(n_{layer}=15, d_{model}=768, n_{heads}=3\) | 55.76 |

#### 离线主结果

| Model | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|---|---:|---:|---:|---:|
| DNN-SIM | 0.7432 | 0.6679 | 0.8737 | 0.6504 |
| MoE-SIM | 0.7484 | 0.6698 | 0.8750 | 0.6519 |
| MultiEmbed-SIM | 0.7501 | 0.6715 | 0.8766 | 0.6525 |
| Wukong-SIM | 0.7568 | 0.6759 | 0.8800 | 0.6530 |
| UserTower-SIM | 0.7593 | 0.6792 | 0.8815 | 0.6550 |
| UserTower-E2E | 0.7576 | 0.6787 | 0.8818 | 0.6548 |
| MTGR-small | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| MTGR-medium | 0.7645 | 0.6843 | 0.8849 | 0.6625 |
| MTGR-large | 0.7661 | 0.6865 | 0.8862 | 0.6646 |

结论：

- MTGR-small 已超过最强 DLRM baseline；
- MTGR-medium / large 继续提升；
- 说明 MTGR 有稳定 scaling 能力；
- UserTower-E2E 反而不如 UserTower-SIM，说明传统 DLRM 复杂度不足以端到端建模长序列，容易 underfitting。

#### 消融实验

| Model | CTR AUC | CTR GAUC | CTCVR AUC | CTCVR GAUC |
|---|---:|---:|---:|---:|
| MTGR-small | 0.7631 | 0.6826 | 0.8840 | 0.6603 |
| w/o cross features | 0.7495 | 0.6689 | 0.8736 | 0.6514 |
| w/o GLN | 0.7606 | 0.6809 | 0.8826 | 0.6585 |
| w/o dynamic mask | 0.7620 | 0.6810 | 0.8828 | 0.6587 |

最重要结论：

> 去掉 cross features 后，效果大幅下降，甚至抹掉 MTGR-large 相对 DLRM 的收益。

这验证了论文核心判断：

> 工业推荐不能简单照搬“去特征工程化”的生成式推荐；cross features 仍然是基础能力。

### 局限性

1. **强依赖工业 cross features**  
   MTGR 的优势很大一部分来自保留和利用 cross features。对于没有丰富 cross features 的公开数据或轻量推荐场景，收益可能不同。

2. **系统工程复杂度高**  
   动态哈希表、all-to-all embedding 通信、动态 batch、专用 kernel、pipeline 训练都需要强工程能力，不是纯模型结构即可复现。

3. **不是典型 LLM 式“生成 item token”**  
   虽然论文称 generative recommendation，但 MTGR 在排序阶段仍是对 candidate token 输出 logits，更像 unified tokenized ranking encoder，而不是自回归生成 item 序列。

4. **在线收益细节在给定材料中不完整**  
   摘要称取得近两年最大在线收益并已承接主流量，但 excerpt 未提供完整在线实验表格和具体业务指标数值。

5. **候选集合仍来自上游召回 / 粗排**  
   MTGR 主要解决 ranking 阶段扩展问题，不等价于端到端替代召回、粗排、精排全链路。

### 放进大模型基础知识体系里怎么理解

#### 1. 它是推荐系统里的“tokenization + scaling”尝试

LLM 的基础范式是：

\[
\text{raw input} \rightarrow \text{tokens} \rightarrow \text{Transformer} \rightarrow \text{next token / logits}
\]

MTGR 类似地把推荐输入改成：

\[
\text{user/item/cross/sequence features} \rightarrow \text{feature tokens} \rightarrow \text{HSTU encoder} \rightarrow \text{candidate logits}
\]

它的意义是：

> 把推荐系统从 feature-specific modules 推向 unified token modeling。

#### 2. 它说明推荐大模型不能盲目去掉特征工程

NLP 里 token 通常已经是统一符号空间；但推荐系统不是。

推荐特征有：

- ID 类稀疏特征；
- 连续统计特征；
- 用户画像；
- 长短期行为序列；
- 实时行为；
- 人工 cross features。

MTGR 的经验是：

> 推荐系统的大模型化不是简单“端到端替代特征工程”，而是要把工业有效特征纳入可扩展架构。

#### 3. 它把 attention 的价值从“序列建模”扩展到“候选交互建模”

传统 target attention 是：

\[
\mathrm{Attention}(candidate, history)
\]

MTGR 是：

\[
\mathrm{SelfAttention}(user, history, realtime, candidates, cross)
\]

这让候选 token 可以在统一上下文中吸收用户、历史、实时行为和 cross feature 信息。

#### 4. 它体现了推荐系统 scaling law 的特殊约束

推荐 scaling 不只看训练 FLOPs，还要看线上延迟。

LLM 可以大量算一个 prompt；推荐排序通常要在毫秒级对大量候选打分。

因此 MTGR 的关键不是“堆大模型”，而是：

\[
\text{重排数据结构} + \text{复用用户计算} + \text{mask 防泄漏} + \text{系统优化}
\]

### 我需要记住什么

1. **MTGR 解决的核心问题**：  
   推荐排序如何在保留工业 cross features 的同时实现大模型 scaling。

2. **传统 DLRM 的瓶颈**：  
   cross module 对每个候选重复计算，模型越大，推理成本随候选数线性爆炸。

3. **MTGR 的核心改造**：  
   把同一用户的多个候选聚合成一个 token 序列，一次 self-attention 输出多个候选 logits。

4. **cross features 是工业推荐的关键资产**：  
   去掉 cross features 会显著掉点，扩大模型也不能完全补偿。

5. **GLN 的作用**：  
   对不同语义空间的 feature tokens 分组归一化，缓解分布不一致。

6. **Dynamic Masking 的作用**：  
   防止实时行为和多候选聚合带来的未来信息泄漏。

7. **实验主结论**：  
   MTGR-small 已超过最强 DLRM；MTGR-medium / large 继续提升，表现出 scaling law。

8. **方法本质**：  
   MTGR 不是简单的“推荐版 LLM”，而是一个面向工业排序约束设计的 **tokenized, attention-based, cross-feature-preserving ranking framework**。
