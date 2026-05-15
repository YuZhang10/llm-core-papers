## Expert-Choice Routing Enables Adaptive Computation in Diffusion Language Models

### 一句话定位

这篇论文研究的是 **DLM-MoE 里的 routing 不应该直接照搬自回归 MoE LLM 常用的 Token-Choice，而应该利用 Diffusion Language Model 每一步能看到全序列的特点，改用 Expert-Choice，并进一步按 mask ratio 动态分配 expert capacity**。

它的核心观点是：

> MoE routing 要和生成范式一起设计。AR LM 适合 Token-Choice，不代表 DLM 也适合 Token-Choice。

---

### 基本信息

- **论文**：Expert-Choice Routing Enables Adaptive Computation in Diffusion Language Models
- **arXiv**：2604.01622
- **时间**：2026-04-02
- **作者**：Shuibai Zhang, Caspian Zhuang, Chihan Cui, Zhihan Yang, Fred Zhangzhi Peng, Yanxin Zhang, Haoyue Bai, Zack Jia, Yang Zhou, Guanhua Chen, Ming Liu
- **代码**：https://github.com/zhangshuibai/EC-DLM
- **关键词**：
  - Diffusion Language Model, DLM
  - Mixture-of-Experts, MoE
  - Token-Choice Routing
  - Expert-Choice Routing
  - Adaptive Computation
  - Mask Ratio
- **引用量记录**：OpenAlex 在 2026-05-11 查询到 `cited_by_count = 0`。Semantic Scholar API 当时被限流，未能交叉确认。考虑到论文刚发布不久，引用数很可能还没有稳定进入各索引。

---

### 背景：DLM 和 AR LM 的差异

#### Autoregressive LM

自回归语言模型按从左到右生成：

$$
p(x) = p(x_1)p(x_2|x_1)\cdots p(x_L|x_{<L})
$$

它每一步只生成下一个 token，不能看到未来 token。所以在 MoE routing 中，让当前 token 自己选 expert 是自然的。

这就是 **Token-Choice, TC**：

> 每个 token 根据 router score 独立选择 top-k expert。

#### Diffusion Language Model

DLM 使用 mask / denoising 过程生成文本。它不是一个 token 一个 token 往右写，而是从大量 `[MASK]` 开始，多步恢复完整序列。

例如：

```text
高 mask ratio: [MASK] [MASK] [MASK] [MASK] 公司
中 mask ratio: 我 [MASK] 今天 [MASK] 公司
低 mask ratio: 我 今天 去 公司 [MASK]
```

DLM 每个 denoising step 都能对整个序列做 bidirectional attention，因此它天然拥有“全局 token 视野”。这正好适合 **Expert-Choice, EC**：

> 不是 token 选 expert，而是每个 expert 从所有 token 中选固定数量的 top-c token。

---

### 研究问题

这篇论文问的是：

> 既然 DLM 每一步都能看到全序列，为什么 DLM-MoE 还要继续使用为自回归模型设计的 Token-Choice routing？能不能改成 Expert-Choice，并进一步让每个 denoising step 使用不同数量的 expert computation？

这里有两层问题：

1. **routing paradigm 问题**：DLM-MoE 应该用 TC 还是 EC？
2. **compute allocation 问题**：DLM 的不同 denoising step 是否应该使用同样多的 expert capacity？

---

### TC vs EC

#### Token-Choice Routing

TC 的选择方向是：

```text
token -> expert
```

每个 token 独立选择自己最喜欢的 expert。

优点：

- 适合 AR LM，因为生成时没有未来 token 的全局视野。
- 工程生态成熟。

问题：

- 热门 expert 会被大量 token 选择，冷门 expert 可能空闲。
- 需要 load-balancing loss，但没有硬保证。
- 在 expert parallelism 下，最忙的 GPU 会成为 straggler，其他 GPU 等待，吞吐下降。

#### Expert-Choice Routing

EC 的选择方向是反过来的：

```text
expert -> token
```

每个 expert 从所有 token 中选择固定数量的 top-c token。

优点：

- 每个 expert 处理固定数量 token，负载均衡由机制保证。
- 不需要额外 load-balancing loss。
- capacity `c` 是外部可控超参数，可以直接控制每个 expert 的计算量。

限制：

- EC 需要看到一批 token 的全局 router score。
- 因此不适合严格逐 token 的 AR inference，但天然适合每一步处理全序列的 DLM。
- EC 保证的是 **expert 负载均衡**，不是天然保证 **每个 token 都被 routed expert 选中**。这个 coverage 问题需要 shared experts、多层路由或更复杂的 constrained assignment 来缓解。

---

### 主要贡献

1. **证明 EC 更适合 DLM-MoE**

EC 通过机制保证 expert 负载均衡，改善 GPU 利用率和训练吞吐。

2. **提出 Timestep-Adaptive Expert Capacity**

利用 EC capacity 可控的特点，让每个 denoising timestep 的 expert capacity 随 mask ratio 动态变化。

3. **发现低 mask-ratio 阶段最值得加计算**

在平均 FLOPs 匹配的条件下，把更多 capacity 分给低 mask-ratio denoising steps 的 Linear-Reverse scheduler 表现最好。

4. **给出机制解释**

低 mask-ratio contexts 的 convergence rate 更高，额外计算的边际收益更大。

5. **验证已有 TC DLM 可以低成本改造成 EC DLM**

只替换 router，保留 expert、embedding 等其他参数，再 finetune，可以获得更快 convergence、更快 decoding 和更高平均准确率。

---

### 核心方法：Timestep-Adaptive Expert Capacity

#### 什么是 mask ratio？

`mask ratio` 是当前 denoising step 中，序列里还有多少比例的 token 是 `[MASK]`。

例如长度为 10 的句子：

```text
[MASK] [MASK] 今天 [MASK] 去 了 [MASK] [MASK] 吗 [MASK]
```

如果有 6 个 token 还是 `[MASK]`，那么：

$$
\text{mask ratio} = \frac{6}{10} = 60\%
$$

高低含义：

- **高 mask ratio**：大部分 token 未知，上下文少。
- **低 mask ratio**：大部分 token 已经恢复，只剩少数 token 要精修。

#### 为什么要分配计算力？

DLM 的生成过程有多个 denoising step，不同 step 的信息状态不同。

高 mask-ratio 阶段：

```text
[MASK] [MASK] [MASK] [MASK] [MASK]
```

这时上下文太少，多给计算也可能只是更认真地猜。

低 mask-ratio 阶段：

```text
我 今天 去 公司 开 [MASK]
```

这时句子已经基本成形，模型更知道剩下位置应该如何补全。额外 expert 更可能转化为更精确的预测。

所以这篇论文不是说“低 mask-ratio 更难”，而是说：

> 低 mask-ratio 阶段的额外计算边际收益更高。

#### 怎么分配计算力？

EC 里每个 expert 处理多少 token 是可控的。这里有两个容易混的符号：

- `c`：每个 expert 真正选择多少 token，也就是 per-expert capacity。
- `k`：等效平均每个 token 使用多少个 routed experts，也就是 compute budget / active expert budget。

假设：

```text
N = 当前 batch/sequence 中的 token 数
E = routed experts 数量
k = 等效每个 token 平均使用的 routed expert 数
c = 每个 expert 选择的 token 数
```

那么 EC 里通常有：

$$
c = \frac{kN}{E}
$$

因为总 token-expert pairs 是：

$$
E \cdot c = kN
$$

所以：

```text
TC: 每个 token 精确选择 k 个 experts
EC: 总共有 kN 个 expert slots，平均到每个 token 是 k 个 slots
```

这意味着，EC 中单个 token 可能被 0 个、1 个、2 个或更多 routed experts 选中；`k` 只表示平均计算预算，不保证每个 token 精确被选 `k` 次。

论文把动态预算写成 capacity function `k(r)`，其中 `r` 是 mask ratio：

$$
k(r) =
\mathrm{clamp}\left(
k_{\min} + (k_{\max} - k_{\min}) \cdot s(r),
k_{\min},
k_{\max}
\right)
$$

直观理解：

```text
k 小 -> 总 expert slots 少 -> 每个 expert 少看一些 token -> 计算量小
k 大 -> 总 expert slots 多 -> 每个 expert 多看一些 token -> 计算量大
```

Linear-Reverse scheduler 使用：

$$
s(r) = 1 - r
$$

也就是：

```text
高 mask ratio -> r 大 -> k 小 -> 少分配 expert capacity
低 mask ratio -> r 小 -> k 大 -> 多分配 expert capacity
```

#### Static EC vs Dynamic EC

**Static EC** 和 **Dynamic EC** 都是 Expert-Choice routing，区别只在于 capacity 是否随 mask ratio 变化。

Static EC 是固定 capacity：

```text
高 mask ratio -> k = 8
中 mask ratio -> k = 8
低 mask ratio -> k = 8
```

也就是每个 denoising step 都用同样的计算预算。

Dynamic EC 是动态 capacity：

```text
高 mask ratio -> k 小
中 mask ratio -> k 中等
低 mask ratio -> k 大
```

论文主要使用 Linear-Reverse scheduler。比如 8B-A1B 实验中：

```text
Static EC:  k = 8
Dynamic EC: k = 2 到 14，平均 k = 8
```

注意这里平均 FLOPs 是匹配的，所以 Dynamic EC 不是靠“总算力更多”赢，而是靠“把同样平均算力重新分配到更值得的 denoising steps”。

---

### 我们讨论中澄清过的几个点

#### 1. 你的复述，修正版

比较准确的版本是：

> 在 EC 中，高 mask-ratio 的时候每个 expert 少看一些 token；低 mask-ratio 的时候每个 expert 多看一些 token。因为论文观察到低 mask-ratio 阶段的学习收敛效率更高，额外计算更划算。

注意不是“低 mask-ratio 收敛效率低所以要补偿”，而是：

> 低 mask-ratio 收敛效率高，所以值得加码。

#### 2. 为什么靠后的 step 还有高 mask ratio？

这里要区分两种 step。

**Denoising step** 是一次生成轨迹里的去噪步骤：

```text
早期 denoising step -> 高 mask ratio
中期 denoising step -> 中 mask ratio
后期 denoising step -> 低 mask ratio
```

如果说的是一次 generation / denoising trajectory，那么靠后的 denoising step 通常应该是低 mask ratio。

但论文 Mechanistic Analysis 图里的 step / stage 很多时候指的是 **training step**，也就是模型训练到多少步之后再做分析。训练时每个 batch 仍然会随机采样不同 mask ratio 的样本：

```text
training step 10000:
  sample A: mask ratio = 0.8
  sample B: mask ratio = 0.2
  sample C: mask ratio = 0.5

training step 200000:
  sample A: mask ratio = 0.9
  sample B: mask ratio = 0.1
  sample C: mask ratio = 0.6
```

所以：

> 靠后的 training step 仍然有高 mask-ratio 样本；靠后的 denoising step 通常是低 mask ratio。

这个问题本质上是 `training step` 和 `denoising step` 两个词撞车了。

#### 3. EC 如何保证每个 token 都被选到？

严格说，EC 本身不保证每个 token 都被 routed expert 选中。

EC 保证的是：

```text
每个 expert 处理固定数量 c 个 token
```

但它不保证：

```text
每个 token 至少被一个 routed expert 处理
```

例如 3 个 experts，每个 expert 选 2 个 tokens：

```text
E1 选: t1, t2
E2 选: t1, t3
E3 选: t1, t4
```

expert 负载是均衡的：

```text
E1: 2 个
E2: 2 个
E3: 2 个
```

但 token 覆盖不是均匀的：

```text
t1 被选了 3 次
t5, t6 一次都没被选
```

所以 EC 的核心 trade-off 是：

```text
TC: 每个 token 有固定 top-k 路由，但 expert 负载可能不均衡
EC: expert 负载均衡，但 token coverage 不天然保证
```

这篇论文用两个机制缓解这个问题：

1. **Shared experts**

模型里有 shared experts，它们无条件处理所有 token。即使某个 token 没被 routed expert 选中，它也仍然会经过 shared expert FFN。

所以：

```text
没被 routed expert 选中 != token 完全没被处理
```

2. **多层 MoE 叠加**

一个 token 在某一层没被选中，不代表它在所有层都没被选中。论文 appendix 统计：

- Static EC，`k=8`：中间层 token drop ratio 小于 1.1%，全层平均约 2.7%。
- Dynamic EC，`k=2-14`：平均 drop ratio 约 8.0%，因为低 capacity step 更容易留下未 routed token。
- 第 0 层 drop ratio 较高，约 20-32%，作者认为是第一层 router 还没形成稳定 token-expert affinity。

作者进一步用近似独立假设估算，一个 token 在所有 16 层都没被 routed expert 处理的概率极小；实践中基本不会出现 token 在全网络深度里完全没被 routed expert 处理的情况。

#### 4. 如果想硬保证 token coverage，有没有相关 routing？

如果目标是同时满足：

```text
每个 token 至少被 1 个 expert 处理
每个 expert 负载尽量均衡
```

那就不再是纯 EC，而更接近 **balanced assignment / optimal transport / min-cost flow routing**。

相关路线：

- **BASE Layers, ICML 2021**：把 token-to-expert allocation 建模成 balanced assignment / linear assignment，让 expert 接收均衡数量的 tokens。它和“每个 token 有分配 + expert 负载均衡”的目标很接近。论文链接：https://proceedings.mlr.press/v139/lewis21a.html
- **Sinkhorn / S-BASE / Optimal Transport routing**：用 Sinkhorn / optimal transport 近似求平衡分配，在 Megatron 等工程框架里也作为 MoE load balancing 选项出现。Megatron 文档：https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/api-guide/moe.html
- **MaxScore, ACL Findings 2025**：把 MoE routing 建成 minimum-cost maximum-flow 问题，目标是减少 capacity constraint 带来的 token dropping 和 padding 浪费。论文链接：https://aclanthology.org/2025.findings-acl.653/

我之前提到的“先给每个 token 一个 top-1 expert 保底，再让 EC 填剩余 capacity”可以看作一个简单工程 heuristic；更正统的学术表述是：

> 带 token 需求约束和 expert 容量约束的 bipartite matching / min-cost flow routing。

---

### 关键实验结果

#### 1. EC vs TC：训练吞吐与负载均衡

论文做了受控预训练实验：架构、数据、超参相同，只改变 routing。

| Routing | Throughput (TFLOP/s/GPU) | Relative to EC |
|---|---:|---:|
| EC | 52.1 | 1.00x |
| TC (cap=1.0) | 35.4 | 0.68x |
| TC (cap=1.25) | 27.0 | 0.52x |
| TC (cap=1.5) | 25.9 | 0.50x |
| TC (dropless) | 24.9 | 0.48x |

EC 达到 loss 3.75 需要 **10.6h**，TC 变体大约需要 **20h**，约 **2x wall-clock 更快**。

机制解释：

- TC 下不同 expert / GPU 负载不均衡，会出现 straggler。
- EC 每个 expert 固定处理相同数量 token，从机制上保证负载均衡。
- 论文的 GPU memory snapshot 中，TC 的显存标准差约 3.6GB，EC 的标准差为 0.0GB。

对应证据：

- Figure 1：training loss vs wall-clock time。
- Table 1：training throughput。
- Figure 2 / routing overview：TC vs EC routing + GPU memory snapshot。

#### 2. Scheduler 对比：多给低 mask-ratio 最好

OpenWebText 上，训练 30B tokens，所有 scheduler 匹配平均 FLOPs。

| Scheduler | $s(r)$ | Compute bias | Final PPL ↓ |
|---|---|---|---:|
| Linear-Reverse | $1-r$ | Low mask ratio | **36.5** |
| Static | -- | -- | 37.1 |
| Cosine-Reverse | $\frac{1}{2}(1+\cos \pi r)$ | Low mask ratio | 37.2 |
| Gaussian | centered at 0.5 | Intermediate | 37.3 |
| Linear | $r$ | High mask ratio | 37.5 |
| Gaussian-Reverse | inverse Gaussian | Extremes | 37.6 |
| Cosine | $\frac{1}{2}(1-\cos \pi r)$ | High mask ratio | 37.6 |

结论：

- 偏向低 mask-ratio 的 Linear-Reverse 最好。
- 同样偏向低 mask-ratio 的 Cosine-Reverse 也不错，但略差于 Linear-Reverse。
- 偏向高 mask-ratio 的 Linear / Cosine 更差。

对应证据：

- Table 2：Scheduling strategies and final validation perplexity。
- Figure 4：Scheduler comparison on OpenWebText。

#### 3. 机制分析：为什么低 mask-ratio 值得加计算？

论文把 mask ratio 分成 4 个区间：

```text
[0, 0.25)      低 mask ratio
[0.25, 0.5)
[0.5, 0.75)
[0.75, 1.0)    高 mask ratio
```

然后统计每个 bin 的 validation loss 下降速度，定义 convergence rate：

$$
\eta_r = -\frac{d \ln \mathcal{L}_r}{dt}
$$

其中：

- $\mathcal{L}_r$：mask-ratio bin `r` 上的 validation loss；
- $t$：training step；
- $\eta_r$ 越大，表示 loss 按比例下降得越快，学习效率越高。

证据在 **Mechanistic Analysis of learning efficiency** 那张图：

- **左图**：静态 EC baseline 下，不同 mask-ratio bin 的 $\eta_r$。低 mask-ratio bin 的 convergence rate 明显最高，论文正文说可达高 mask-ratio bin 的约 7x，caption / summary 中进一步描述为一个数量级以上。
- **右图**：Dynamic EC 相比 Static EC 的 $\eta_r^{dyn}/\eta_r^{static}$。大于 1 表示 Dynamic EC 学得更快。提升主要集中在低 mask-ratio bins，高 mask-ratio bins 反而可能略低。

所以证据链是：

```text
Figure Mechanistic Analysis 左图:
低 mask-ratio 本来学习效率最高

Figure Mechanistic Analysis 右图:
Dynamic EC 的额外收益主要发生在低 mask-ratio

Table 2 + Figure 4:
把更多 capacity 分给低 mask-ratio 的 scheduler 结果最好
```

这就是“低 mask-ratio 收敛效率更高，所以值得加码”的实验依据。

#### 4. 8B-A1B scaling validation

论文在 8B total parameters / 1B active parameters 的 DLM-MoE 上验证：

- 数据：Nemotron-CC
- Static EC：固定 $k=8$，每个 denoising step 使用相同平均 expert budget。
- Dynamic EC：Linear-Reverse，$k_{\min}=2, k_{\max}=14$，高 mask-ratio 少算、低 mask-ratio 多算，平均 $k=8$。
- 平均 FLOPs 匹配

结果：

- validation perplexity 更好；
- MMLU 5-shot 更好；
- ARC-Challenge 25-shot 更好；
- 每个 checkpoint 上 Dynamic EC 都优于 Static EC。

对应证据：

- Figure 5：8B-A1B pretraining comparison。

#### 5. Retrofitting：已有 TC DLM 能否直接改 EC？

论文对已有 LLaDA-MoE 做 router replacement：

- 替换 token-choice gate 为 expert-choice gate；
- 保留 expert、embedding 和其他参数；
- 在 GSM8K、HumanEval、HumanEval+、MedQA 上 finetune。

汇总结果：

| Routing | Avg Accuracy | Avg Decode Time |
|---|---:|---:|
| TC | 52.6 | 1324s |
| EC | 53.6 | 962s |
| Dynamic EC | **54.9** | 988s |

结论：

- EC / Dynamic EC 的 decoding 更快，约 1.3x-1.5x。
- Dynamic EC 的平均精度最高。
- 已有 TC DLM 可以通过只换 router 的方式低成本获得收益。

对应证据：

- Figure 6：retrofitting SFT results。
- Table 3：peak accuracy and evaluation decode time。

---

### 我对这篇论文的理解

这篇工作的价值不只是“EC 比 TC 快”，而是把 DLM 的生成机制和 MoE 的稀疏计算重新对齐了。

在 AR LM 里，TC 是自然选择，因为模型生成时没有未来 token，不能全局安排 expert。但 DLM 每个 denoising step 都处理完整序列，所以它满足 EC 所需要的全局 token score 条件。

更进一步，DLM 的 timestep / mask ratio 本身就是一个天然的 adaptive computation 轴：

```text
不同 denoising step 的任务状态不同
-> 不应该每一步都用同样多 expert capacity
-> EC 刚好让 capacity 可控
-> 可以把计算从边际收益低的阶段挪到边际收益高的阶段
```

这点比单纯的 routing 替换更有启发性。

---

### 局限与可以继续想的方向

1. **scheduler 是手工设计的**

Linear-Reverse、Cosine-Reverse、Gaussian 等都是人工指定函数。最优 scheduler 可能依赖模型规模、数据分布、任务类型和训练阶段。

2. **低 mask-ratio 加码不一定处处成立**

论文在当前设置中观察到低 mask-ratio 边际收益更高，但如果数据、mask schedule、模型结构或任务形式改变，最优分配可能变化。

3. **EC 可能带来 token coverage 问题**

因为是 expert 选择 token，可能有些 token 没有被 routed expert 选中。论文用 shared experts 和多层路由缓解，但如果需要每层硬保证 coverage，就可能要引入 balanced assignment / optimal transport / min-cost flow 这类更复杂 routing。

4. **Linear-Reverse vs Cosine-Reverse 的差异还没完全解释清楚**

二者都偏向低 mask-ratio，Cosine-Reverse 更激进。论文认为 Cosine-Reverse 可能过度饿死高 mask-ratio 阶段；但 appendix 中也提到 per-bin convergence rate 未必能完全解释二者 PPL 差距。

5. **自然扩展：learned scheduler**

可以让模型自动学习每个 timestep 应该使用多少 expert capacity，例如：

- 根据 mask ratio 学 capacity；
- 根据模型置信度 / uncertainty 学 capacity；
- inference-time adaptive compute；
- 用 RL 或 lightweight predictor 学动态策略。

---

### 最短复习版

这篇论文可以记成三句话：

1. **DLM-MoE 更适合 Expert-Choice，而不是 Token-Choice**，因为 DLM 每一步能看到全序列，EC 可以硬保证 expert 负载均衡。
2. **论文里的 `k` 是平均 expert budget；EC 真正固定的是每个 expert 选多少 token `c = kN/E`**，所以 EC 保证负载均衡，但不天然保证每个 token 每层都被 routed expert 选中。
3. **Dynamic EC 按 timestep / mask ratio 调整 `k`**：高 mask-ratio 少算、低 mask-ratio 多算；实验发现低 mask-ratio 阶段学习效率更高、额外计算更划算，所以 Linear-Reverse scheduler 在平均 FLOPs 匹配下表现最好。
