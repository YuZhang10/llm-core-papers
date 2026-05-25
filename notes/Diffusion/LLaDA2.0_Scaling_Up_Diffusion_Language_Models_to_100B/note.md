---
title: "LLaDA2.0: Scaling Up Diffusion Language Models to 100B"
short_title: "LLaDA2.0"
paper_id: "arXiv:2512.15745"
authors: "Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, Chengxi Li, Chongxuan Li, Jianguo Li, Zehuan Li, Huabin Liu, Lin Liu, Guoshan Lu, Xiaocheng Lu, Yuxin Ma, Jianfeng Tan, Lanning Wei, Ji-Rong Wen, Yipeng Xing, Xiaolu Zhang, Junbo Zhao, Da Zheng, Jun Zhou, Junlin Zhou, Zhanchao Zhou, Liwang Zhu, Yihong Zhuang"
venue: "arXiv 2025"
date: "2026-05-18"
tags:
  - diffusion
  - discrete-diffusion
  - language-model
  - block-diffusion
  - MoE
  - LLaDA
---

# LLaDA2.0: Scaling Up Diffusion Language Models to 100B

## 本地文件

- PDF: [LLaDA2.0_Scaling_Up_Diffusion_Language_Models_to_100B_arXiv_2512.15745.pdf](LLaDA2.0_Scaling_Up_Diffusion_Language_Models_to_100B_arXiv_2512.15745.pdf)
- arXiv 源码包: [LLaDA2.0_arXiv_2512.15745_source.tar.gz](LLaDA2.0_arXiv_2512.15745_source.tar.gz)
- arXiv 源码目录: [LLaDA2.0_arXiv_2512.15745_source](LLaDA2.0_arXiv_2512.15745_source)
- 官方代码 / 模型仓库: https://github.com/inclusionAI/LLaDA2.X
- Hugging Face collection: https://hf.co/collections/inclusionAI/llada-20
- arXiv: https://arxiv.org/abs/2512.15745

## 一句话理解

LLaDA2.0 的核心不是重新发明 masked diffusion，而是回答一个工程化 scaling 问题:

> 既然从头训练 diffusion LLM 很贵，那能不能从一个已经很强的 autoregressive model 出发，把它稳定转换成 diffusion / block diffusion language model，并且扩到 100B 级别？

它给出的答案是: 可以。关键 recipe 是:

```text
强 AR checkpoint
  -> 用 Warmup-Stable-Decay 做 continual pre-training
  -> 逐步转换成 masked diffusion / block diffusion model
  -> 用 SFT + DPO + CAP 做 post-training
  -> 得到 16B mini 和 100B flash 两个 MoE diffusion LLM
```

所以 LLaDA2.0 和 LLaDA 的关系可以这样看:

```text
LLaDA:
    从头训练 8B masked diffusion language model
    证明 diffusion LLM 可以长出 LLM-like capabilities

LLaDA2.0:
    从强 AR 模型转换到 diffusion / block diffusion
    证明这条路线可以 scale 到 100B，并且更接近可部署系统
```

## 论文主线

### 1. 这篇论文解决的是 diffusion LLM 的 scaling 问题

LLaDA 已经说明:

```text
离散文本 token 也可以做 diffusion:
随机 mask -> 学会补 mask -> 从全 mask 逐步生成文本
```

但 LLaDA 的路线有一个现实问题: 从头训练一个大 diffusion LLM 成本很高，而且 AR 模型的训练基础设施、数据配方、scaling 经验已经成熟很多。

LLaDA2.0 选择了一条更务实的路:

```text
不要完全从头训练 diffusion LLM
先拿一个强 autoregressive model
再把它转换成 diffusion / block diffusion model
```

这背后的判断是:

```text
AR 模型里已经有大量语言知识
diffusion 需要的是改变生成和去噪机制
如果能保留 AR 知识，同时注入 diffusion 能力，就能省掉大量预训练成本
```

### 2. 为什么不能直接把 AR 改成 diffusion？

直接转换很难，因为 AR 和 diffusion 的训练分布不一样:

| 范式 | 模型看到什么 | 学什么 |
|---|---|---|
| AR | 左侧 prefix | 下一个 token |
| MDLM / LLaDA | 任意 masked sequence | 补任意 masked token |
| BDLM | 前面 clean blocks + 当前 noisy block | denoise 当前 block |

如果突然把一个 AR 模型改成 bidirectional denoising，容易出现两个问题:

- 优化不稳定，模型还没适应 mask / bidirectional context。
- 预训练知识遗忘，原来 AR 模型里学到的语言能力被破坏。

LLaDA2.0 的主设计就是避免这个硬切换。

### 3. Block Diffusion 是中间桥梁

LLaDA2.0 使用 Block Diffusion Language Model, BDLM，作为 AR 和 full masked diffusion 之间的桥。

直觉上，BDLM 不是整段 response 一起 diffusion，而是:

```text
block 1 -> block 2 -> block 3 -> ...
```

块与块之间保留近似 autoregressive 的顺序；每个 block 内部用 diffusion 方式反复 denoise。

可以理解为:

```text
AR:
    每次生成 1 个 token

Full MDLM / LLaDA:
    整段 response 一起从 mask 中恢复

BDLM:
    每次生成 1 个 block
    block 内部并行补 token
```

这给了 LLaDA2.0 一个很漂亮的连续谱:

| block size | 近似形态 |
|---:|---|
| 1 | autoregressive，每次一个 token |
| 32 | block diffusion，块内并行 |
| 4096 | full-sequence masked diffusion |

所以 AR 模型可以被看成 block size = 1 的极端情况。LLaDA2.0 正是利用这个视角，让模型慢慢从 AR 过渡到 diffusion。

### 4. WSD: Warmup-Stable-Decay

LLaDA2.0 最核心的训练 recipe 是 WSD:

```text
Warmup:
    block size 从 1 慢慢变大

Stable:
    block size 到 4096，做 full-sequence MDLM 大规模训练

Decay:
    block size 再降回较小值，比如 32，用于高效推理
```

更具体地:

```text
AR base model
  block size = 1
  -> 4
  -> 32
  -> 64
  -> 4096
  -> 大规模 full MDLM training
  -> 4096 -> 2048 -> ... -> 32
```

这套 schedule 的直觉是:

```text
先别一下子把 AR 模型扔进全局 bidirectional denoising
先让它逐步适应更大的块、更强的并行去噪
等它学会 full-sequence denoising 后
再收缩回 block diffusion，换取推理效率和 KV-cache 复用
```

这也是这篇和 LLaDA 原始论文最大的不同之一:

```text
LLaDA:    直接训练 masked diffusion language model
LLaDA2.0: 用 AR -> BDLM -> MDLM -> BDLM 的路径做转换
```

### 5. BDLM / MDLM 的训练目标

在 BDLM 的 warmup 和 decay 阶段，目标是只对当前 noisy block 里的 `[MASK]` token 计算交叉熵:

$$
\mathcal{L}_{\text{BDLM}}(\theta)
=
-
\mathbb{E}_{t,x_0,x_t}
\left[
\frac{\alpha_t'}{1-\alpha_t}
\sum_{k=1}^{K}
\sum_{i=1}^{L_B}
\mathbf{1}[x_{t,k}^i=\text{[MASK]}]
\log p_\theta(x_{0,k}^i | x_{0,<k}, x_{t,k})
\right]
$$

这里可以先抓住几个对象:

- $L_B$: block size。
- $K$: block 数。
- $x_{0,<k}$: 当前 block 之前的 clean blocks。
- $x_{t,k}$: 当前 noisy block。
- $x_{0,k}^i$: 当前 block 里第 $i$ 个原 token。

换成伪代码:

```python
for batch in data:
    x0 = batch.tokens
    t = sample_timestep()
    xt = mask_some_tokens(x0, t)

    for block_k in blocks:
        context = clean_blocks_before_k
        noisy_block = xt[block_k]

        logits = model(context, noisy_block)
        loss += CE(logits[masked_positions], x0[masked_positions])
```

当 block size 变成整个序列长度时，$K=1$，BDLM 就退化成 full-sequence MDLM:

$$
\mathcal{L}_{\text{MDLM}}(\theta)
=
-
\mathbb{E}_{t,x_0,x_t}
\left[
\frac{\alpha_t'}{1-\alpha_t}
\sum_{i=1}^{L}
\mathbf{1}[x_t^i=\text{[MASK]}]
\log p_\theta(x_0^i | x_t)
\right]
$$

这和 LLaDA 原始论文里的 masked diffusion objective 是同一类东西，只是 LLaDA2.0 把它放进了 block diffusion 和 AR 转换的框架里。

### 6. Document-level attention mask

LLaDA2.0 训练时会 packing 多篇文档来提高吞吐。但 full / bidirectional attention 有一个坑:

```text
如果多个无关文档被 pack 到同一个长序列里，
bidirectional denoising 可能让 token 看到另一篇文档，
形成假的上下文依赖。
```

这在 AR 里相对没那么严重，因为 causal mask 本来就限制了信息流。但 diffusion 需要双向看上下文，跨文档污染会更麻烦。

所以 LLaDA2.0 加了 document-level attention mask:

```text
同一文档内部可以 attention
不同文档之间不能 attention
```

在 block diffusion 设置里，它还要同时满足:

- noisy block 内部的 block-diagonal attention。
- 当前 noisy block 对之前 clean blocks 的条件依赖。
- clean blocks 内部的 block-causal 结构。
- 防止 clean tokens 反向看到 noisy tokens。

这部分是很工程但很关键的点: diffusion LLM 一旦做大规模 packed training，attention mask 不是小细节，而是训练稳定性的基础设施。

### 7. Post-training: SFT、complementary masking、CAP、DPO

LLaDA2.0 的 post-training 也不是简单套用 AR 模型流程，而是把常见 alignment 技术改写到 diffusion / block diffusion 的目标上。

#### SFT

SFT 时，模型根据 prompt $c$ 生成 response $x_0$。训练目标是从 noisy response block 里恢复 clean token:

$$
\mathcal{L}_{\text{SFT}}(\theta)
=
-
\mathbb{E}_{t,(c,x_0),x_t}
\left[
\frac{\alpha_t'}{1-\alpha_t}
\sum_{k=1}^{K}
\sum_{i=1}^{L_B}
\mathbf{1}[x_{t,k}^i=\text{[MASK]}]
\log p_\theta(x_{0,k}^i | c, x_{0,<k}, x_{t,k})
\right]
$$

直觉:

```text
prompt c 不被破坏
前面已经生成好的 blocks 是 clean context
当前 block 是 noisy block
模型学习把当前 block 补干净
```

#### Mask ratio bandwidth

普通 masked diffusion 会从 $[0,1]$ 里采样 mask ratio。但极端 mask ratio 学习信号不一定好:

```text
mask 很少:
    任务太容易，梯度信息少

mask 很多:
    上下文太少，接近学 token 边际分布
```

所以 LLaDA2.0 在 SFT 里限制 mask ratio 到一个区间 $[\alpha_{\min}, \alpha_{\max}]$，让训练集中在信息量更高的噪声强度。

#### Complementary masking

Complementary masking 的思想很直观:

```text
对同一个 response 采样一个 mask
再构造它的反 mask
两个样本一起训练
```

如果第一个样本 mask 了位置 A，第二个样本就 mask A 的补集。这样一对样本覆盖了所有 token，保证每个 token 都贡献训练信号。

这解决了 diffusion fine-tuning 的一个问题:

```text
随机 mask 每次只训练一部分 token
样本利用率不满
```

Complementary masking 接近把每条 SFT 样本的 token-level 数据利用率拉到 100%。

#### CAP: Confidence-Aware Parallel

Diffusion 生成想快，就要一次接受多个 token。但如果模型不够自信，就只能小步慢慢 denoise。

LLaDA2.0 加了一个 confidence loss:

```text
如果某个 token 已经预测对了
就鼓励模型对这个正确预测更有把握
```

总目标是:

$$
\mathcal{L}(\theta)
=
\mathcal{L}_{\text{SFT}}(\theta)
+
\lambda \mathcal{L}_{\text{conf}}(\theta)
$$

它的作用不是让模型“知道答案”本身，而是让正确答案的概率分布更尖锐。这样推理时可以更激进地并行接受 token。

#### DPO

DPO 原本需要比较 chosen / rejected response 的 log probability。但 diffusion model 的完整 log-likelihood 不像 AR 那样容易精确计算。

LLaDA2.0 的处理方式是:

```text
用 Block Diffusion 的 ELBO 近似 response log-prob
再在这个 ELBO 上做 DPO
```

也就是说，preference alignment 仍然是:

```text
提高 chosen response 的相对分数
降低 rejected response 的相对分数
```

只是“分数”从 AR log-prob 换成了 diffusion reconstruction / ELBO 形式。

### 8. 推理: block-by-block + threshold acceptance

推理时，LLaDA2.0 不像 LLaDA 原始模型那样整段 response 一起从 full mask 里恢复。它采用 block diffusion:

```text
已经生成的前面 blocks 作为 context
当前 block 从 mask/noisy 状态开始
多步并行 denoise
完成后进入下一个 block
```

每一步会预测当前 block 里所有未填位置，然后用一个混合接受策略:

```text
如果 token 概率超过 threshold:
    接受这个 token

如果接受得太少:
    fallback 接受若干最高置信 token，保证生成继续前进
```

论文主评测使用:

```text
temperature = 0.0
block size = 32
threshold = 0.95
```

block size 和 threshold 是质量 / 速度旋钮。论文的小规模分析里:

| 设置 | 分数 | TPF |
|---|---:|---:|
| threshold 0.95, block size 32 | 70.15 | 2.55 |
| threshold 0.85, block size 32 | 67.90 | 3.31 |
| threshold 0.95, block size 16 | 70.26 | 2.44 |

所以主设置选了 block size 32、threshold 0.95: 质量接近最优，速度明显好于 block size 16。

## 模型版本

论文发布两个 instruction-tuned MoE 版本:

| Model | 参数规模 | 定位 |
|---|---:|---|
| LLaDA2.0-mini | 16B | 资源受限场景 |
| LLaDA2.0-flash | 100B | 高性能场景 |

GitHub / Hugging Face 还提供 CAP 版本:

| Model | 说明 |
|---|---|
| LLaDA2.0-mini-CAP | 加入 Confidence-Aware Parallel 的高效推理版本 |
| LLaDA2.0-flash-CAP | 100B 级 CAP 版本 |

论文用的 AR 起点是:

```text
Ling-mini-2.0
Ling-flash-2.0
```

这也是为什么 LLaDA2.0 的实验要和 Ling 系列、Qwen3 系列对比: 它关心的不是“纯从头训练 diffusion 能不能赢”，而是“AR 初始化后转换出的 diffusion model 能不能接近或超过强 AR peer”。

## 主要实验结论

### 1. mini 接近同级 AR 模型

LLaDA2.0-mini 的 47 项 benchmark 平均分:

| Model | Average |
|---|---:|
| Qwen3-8B no_think | 63.42 |
| Ling-mini-2.0 | 65.77 |
| LLaDA2.0-mini-preview | 54.67 |
| LLaDA2.0-mini | 64.34 |

一些代表性结果:

| Benchmark | Qwen3-8B | Ling-mini-2.0 | LLaDA2.0-mini |
|---|---:|---:|---:|
| MMLU | 80.94 | 82.15 | 80.53 |
| SQuAD 2.0 | 85.21 | 75.56 | 86.50 |
| HumanEval | 84.76 | 85.98 | 86.59 |
| GSM8K | 93.63 | 94.62 | 94.24 |
| IFEval strict prompt | 86.90 | 76.16 | 80.78 |
| BFCL v3 | 70.08 | 53.98 | 70.90 |

mini 的读法是: 平均分还没明显超过 Ling-mini，但已经不像早期 diffusion LLM 那样掉队，并且在代码、部分 reasoning、tool/function calling 上很有信号。

### 2. flash 基本追平强 AR peer，并在结构化任务上有优势

LLaDA2.0-flash 的平均分:

| Model | Average |
|---|---:|
| Qwen3-30B-A3B-Instruct-2507 | 73.60 |
| Ling-flash-2.0 | 72.15 |
| LLaDA2.0-flash-preview | 65.97 |
| LLaDA2.0-flash | 73.18 |

一些代表性结果:

| Benchmark | Qwen3-30B-A3B | Ling-flash-2.0 | LLaDA2.0-flash |
|---|---:|---:|---:|
| MMLU | 87.13 | 87.98 | 87.69 |
| ARC-c | 95.81 | 95.08 | 95.93 |
| HumanEval | 93.29 | 85.98 | 94.51 |
| MBPP | 86.65 | 85.01 | 88.29 |
| MultiPL-E | 70.67 | 65.76 | 74.87 |
| AIME 2025 | 61.88 | 55.89 | 60.00 |
| BFCL v3 | 73.19 | 67.57 | 75.43 |
| Nexus FC | 49.93 | 36.25 | 50.45 |

作者的重点结论是:

```text
LLaDA2.0-flash 在总体平均分上已经接近 Qwen3-30B-A3B，
并且在 coding、agent/tool use、部分 math 上开始展现 diffusion / block diffusion 的优势。
```

这个结论要谨慎读: 它不是所有 benchmark 都赢，也不是说 diffusion 已经全面替代 AR。但它证明了 100B 级 diffusion LLM 不只是能跑，而且可以进入强模型竞争区间。

### 3. CAP 带来推理速度提升

论文的推理速度分析里，LLaDA2.0-flash-CAP 在四个代码和数学 benchmark 上达到:

| Model | TPS |
|---|---:|
| LLaDA2.0-flash-CAP | 535 |
| LLaDA2.0-flash | 383 |
| AR baseline 1 | 256 |
| AR baseline 2 | 237 |

作者报告 CAP 版本相对 AR baseline 可到约 2.1x speed-up。

这里最值得记住的不是具体 TPS，而是机制:

```text
block diffusion 让模型具备并行生成空间
CAP 让模型更敢一次接受多个 token
推理引擎 dInfer / SGLang 支持 block-level 优化和 KV-cache 复用
```

## 几个容易卡住的技术点

### 1. Block Diffusion 是什么？

Block Diffusion 是 AR 和 full diffusion 之间的折中。

普通 AR 是:

```text
一次生成 1 个 token
token1 -> token2 -> token3 -> ...
```

LLaDA 原始 full diffusion 更像:

```text
整段 response 一起从 [MASK] 里恢复
[MASK MASK MASK ...] -> 多步去噪 -> 完整文本
```

Block Diffusion 则是:

```text
一次生成一个 block
block1 -> block2 -> block3 -> ...
每个 block 内部用 diffusion 并行补 token
```

比如 block size = 32 时，模型先生成前 32 个 token，这 32 个 token 内部可以并行预测和反复 denoise；这个 block 稳定后，再生成下一个 32-token block。

所以它可以看成一条连续谱:

```text
block size = 1      约等于 AR
block size = 32     实用 block diffusion
block size = 全序列 约等于 full LLaDA / MDLM
```

LLaDA2.0 用它做两件事:

```text
训练时:
    从 AR 平滑过渡到 diffusion

推理时:
    保留并行生成优势，同时还能复用前面 blocks 的 KV cache
```

### 2. Top-k checkpoint merge 是什么？

Top-k checkpoint merge 是训练结束后选出验证表现最好的 $k$ 个 checkpoint，然后把它们的权重平均成最终模型。

伪代码:

```python
top_ckpts = [ckpt_1, ckpt_2, ckpt_3]

merged_weight = (
    ckpt_1.weight +
    ckpt_2.weight +
    ckpt_3.weight
) / 3
```

它有点像 model soup。直觉是:

```text
单个 checkpoint 可能有点抖
几个好 checkpoint 平均后，参数落在更平滑、更稳的位置
```

它和 EMA 不一样:

| 方法 | 什么时候做 | 做法 |
|---|---|---|
| EMA | 训练过程中 | 每一步滑动平均参数 |
| Top-k checkpoint merge | 训练结束后 | 挑 top-k 个好 checkpoint 离线平均 |

注意，这通常只适合同一次训练轨迹附近的 checkpoint，不能随便拿几个不同来源的模型硬平均。

### 3. Complementary masking 是什么？

Complementary masking 是对同一个样本做一对互补 mask，让所有 token 都能贡献训练信号。

普通 masking:

```text
原句:  A B C D E F

mask:  A [M] C [M] E F
训练:     B     D
```

这一轮只训练了被 mask 的 token，也就是 `B` 和 `D`。

Complementary masking 会再构造一个反 mask:

```text
原句:  A B C D E F

mask1: A [M] C [M] E F
训练:     B     D

mask2: [M] B [M] D [M] [M]
训练:  A     C     E   F
```

合起来，每个 token 都被训练到一次。这解决了 diffusion SFT 里的样本利用率问题:

```text
随机 mask 每次只训练一部分 token
互补 mask 让一条数据几乎 100% 被用起来
```

### 4. CAP 训练是什么？

CAP = Confidence-Aware Parallel。它是为了让 diffusion LLM 推理时更快接受多个 token。

Diffusion 生成时，模型每一步会同时猜很多 token。但如果模型不够自信，就不能一次接受很多 token，只能继续 denoise。CAP 加了一个辅助 confidence loss，让模型对已经预测正确的 token 更有信心。

这里的“已经预测正确”发生在训练时。因为训练时有标准答案，所以可以判断模型当前 top-1 预测是否等于 ground truth。

例如 response 是:

```text
The capital of France is Paris.
```

训练时 mask 成:

```text
The capital of France is [MASK].
```

真实答案是:

```text
[MASK] = Paris
```

如果模型输出:

```text
Paris   0.45
Lyon    0.30
London  0.15
Berlin  0.10
```

top-1 是 `Paris`，真实答案也是 `Paris`，所以这个位置就是“已经预测正确的 token”。但它的信心只有 0.45，推理时还不敢直接定稿。

CAP 会额外鼓励这种已经预测对的位置分布更尖锐，例如变成:

```text
Paris   0.97
Lyon    0.01
London  0.01
Berlin  0.01
```

具体做法是: 对这些 top-1 已经等于真实 token 的位置，最小化输出分布的 entropy。

```python
logits = model(prompt, noisy_block)
probs = softmax(logits)

ce_loss = cross_entropy(logits[masked_pos], target[masked_pos])

pred = probs.argmax(dim=-1)
correct = pred == target

entropy = -(probs * probs.log()).sum(dim=-1)
conf_loss = entropy[masked_pos & correct].mean()

loss = ce_loss + lambda_ * conf_loss
```

为什么只对预测正确的位置做 confidence loss？因为如果预测错了，还鼓励它更自信，就会变成错得更坚定。

例如真实答案是 `Paris`，但模型输出:

```text
Lyon    0.55
Paris   0.30
London  0.10
Berlin  0.05
```

这时如果也最小化 entropy，可能会把 `Lyon` 推到 0.95，反而强化错误。

所以 CAP 的分工是:

```text
SFT loss:
    负责把答案预测对

confidence loss:
    只在已经预测对的位置上，让分布更尖锐
```

### 5. Threshold acceptance 是什么？

Threshold acceptance 是推理时决定哪些 token 可以定稿的规则。

在 block diffusion 生成里，当前 block 很多位置一开始是 `[MASK]`。每一步模型会同时给这些位置预测 token 和概率:

```text
位置1: Paris  0.97
位置2: is     0.91
位置3: known  0.62
位置4: for    0.96
位置5: food   0.48
```

假设 threshold = 0.95，那么:

```text
概率 >= 0.95 的 token 接受 / 固定下来
概率 < 0.95 的 token 继续保持 mask，下一步再预测
```

结果是:

```text
位置1: Paris  接受
位置2: is     不接受，继续 mask
位置3: known  不接受，继续 mask
位置4: for    接受
位置5: food   不接受，继续 mask
```

伪代码:

```python
if max_prob(position) >= threshold:
    accept_token()
else:
    keep_masked()
```

threshold 越高，模型越保守，质量更稳但生成更慢。threshold 越低，模型越激进，一次接受 token 更多，但可能早早固定错误 token。

这也解释了 CAP 为什么有用: CAP 让正确 token 的概率更高，于是更多 token 能跨过 threshold，被更早接受。

### 6. ELBO 是什么？

ELBO = Evidence Lower Bound，证据下界。

生成模型真正想最大化的是:

$$
\log p_\theta(x)
$$

也就是模型给真实样本的概率。但在 diffusion / latent variable model 里，精确的 $\log p_\theta(x)$ 往往很难直接算。

于是我们优化一个可计算的下界:

$$
\text{ELBO}(x) \le \log p_\theta(x)
$$

最大化 ELBO，就等价于尽量提高真实 likelihood 的一个可计算代理。

在 LLaDA2.0 的 DPO 里，问题是:

```text
AR 模型可以直接算 response 的 log probability
diffusion model 的精确 log probability 不好算
```

所以它用 Block Diffusion 的 ELBO 当作 response 分数:

```text
score(response | prompt) ≈ ELBO(response | prompt)
```

然后做 DPO:

```text
让 chosen response 的 ELBO 高于 rejected response 的 ELBO
```

一句话总结:

```text
ELBO 是 diffusion 模型里替代精确 log-likelihood 的可优化代理分数。
```

## 工程基础设施

LLaDA2.0 不是一篇只有算法 idea 的论文，它有很重的系统工程部分:

- 预训练使用 Megatron-LM。
- 100B 训练用 DP、PP、TP、CP、EP 等并行。
- block diffusion attention mask 用 cuDNN backend。
- 训练 LLaDA2.0-mini 时，相比 TransformerEngine unfused attention，attention 层内存节省超过 90%，端到端速度提升超过 1.3x。
- AR 到 diffusion 转换初期会遇到 mask token embedding 范数过小导致的梯度爆炸，论文用在 masked token embedding 输出上加独立高斯噪声来稳定训练。
- post-training 使用 dFactory / VeOmni。
- 推理用 dInfer，并把 block diffusion inference 接到 SGLang 的优化路径上。

这一部分说明: diffusion LLM 到 100B 以后，问题不只是 objective，也包括 attention mask、并行策略、推理引擎和数值稳定。

## 和 LLaDA 的连接

可以把两篇放成这样:

| 论文 | 核心问题 | 路线 |
|---|---|---|
| LLaDA | diffusion LLM 是否能从头训练出 LLM-like capabilities？ | 8B masked diffusion from scratch |
| LLaDA2.0 | diffusion LLM 如何 scale 到 100B 并接近部署？ | AR checkpoint conversion + WSD + BDLM + MoE |

LLaDA2.0 继承了 LLaDA 的 masked reconstruction 思想，但把它变成了更实用的 scaling recipe:

```text
不是“从全 mask 生成整段文本”这一种形态，
而是把 AR、block diffusion、full MDLM 看成一条连续谱。
```

这也是它最重要的思想贡献。

## 和 DDPM / DiT 的连接

在 diffusion 脉络里，LLaDA2.0 的位置更像:

```text
DDPM:
    证明连续图像上逐步去噪可行

DiT:
    证明 Transformer 可以做图像 diffusion denoiser，并且有 scaling behavior

LLaDA:
    证明离散文本 token 可以做 masked diffusion，并能 scale 到 8B

LLaDA2.0:
    证明 diffusion language model 可以借助 AR 初始化和 block diffusion 扩到 100B
```

如果说 LLaDA 是“范式存在性证明”，LLaDA2.0 更像是“规模化工程路线图”。

## 需要记住的点

1. LLaDA2.0 不是从头训练，而是从 AR checkpoint 转换成 diffusion / block diffusion。
2. 它把 AR 看成 block size = 1 的特殊 BDLM。
3. WSD 是核心: block size 逐渐增大到 4096，再降回小 block。
4. Stable 阶段用 full-sequence MDLM 学全局 denoising。
5. Decay 阶段回到小 block，是为了推理效率、变长生成和 KV-cache 复用。
6. Document-level attention mask 解决 packed training 下跨文档污染。
7. SFT 里使用 mask ratio bandwidth 和 complementary masking 提高稳定性和数据利用率。
8. CAP 通过 confidence loss 让模型更适合并行接受 token。
9. DPO 被改写到 diffusion / block diffusion 的 ELBO 分数上。
10. LLaDA2.0-mini 是 16B，LLaDA2.0-flash 是 100B MoE。
11. flash 在平均分上接近 Qwen3-30B-A3B，并在代码、agent、部分数学任务上有优势。
12. 这篇真正重要的是从“diffusion LLM 可行”推进到“diffusion LLM 可规模化、可部署”。

## 我的理解

LLaDA2.0 最有启发的地方，是它没有把 AR 和 diffusion 看成非此即彼。

它把两者放在一个 block size 控制的连续空间里:

```text
block size = 1:
    几乎就是 AR

block size = 32:
    实用 block diffusion

block size = 4096:
    full masked diffusion
```

这样一来，问题就从:

```text
AR 和 diffusion 哪个是未来？
```

变成:

```text
在训练、推理、长上下文、结构化任务之间，
我们应该选哪个 block size、哪个 attention mask、哪个 post-training 目标？
```

这比单纯争论“next-token prediction 会不会被替代”更有建设性。LLaDA2.0 的意义就在这里: 它把 diffusion LLM 从一个有趣范式，往可扩展系统推进了一大步。
