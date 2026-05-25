---
title: "Large Language Diffusion Models"
short_title: "LLaDA"
paper_id: "arXiv:2502.09992"
authors: "Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li"
venue: "NeurIPS 2025"
date: "2026-05-18"
tags:
  - diffusion
  - discrete-diffusion
  - language-model
  - masked-diffusion
  - non-autoregressive
---

# Large Language Diffusion Models

## 本地文件

- PDF: [Large_Language_Diffusion_Models_LLaDA_arXiv_2502.09992.pdf](Large_Language_Diffusion_Models_LLaDA_arXiv_2502.09992.pdf)
- arXiv 源码包: [LLaDA_arXiv_2502.09992_source.tar.gz](LLaDA_arXiv_2502.09992_source.tar.gz)
- arXiv 源码目录: [LLaDA_arXiv_2502.09992_source](LLaDA_arXiv_2502.09992_source)
- 论文主页: https://ml-gsai.github.io/LLaDA-demo/
- 官方代码: https://github.com/ML-GSAI/LLaDA
- arXiv: https://arxiv.org/abs/2502.09992

## 一句话理解

LLaDA 想回答一个很大胆的问题:

> LLM 的核心能力一定要靠 left-to-right next-token prediction 吗？还是说，只要还是一个足够好的 generative model，diffusion 也能长出 scaling、in-context learning、instruction following 这些能力？

它给出的路线是: 不再像 GPT 一样从左到右预测下一个 token，而是把文本序列随机 mask 掉一部分，让 Transformer 学会同时预测所有 masked tokens。生成时从一串全 `[MASK]` 的 response 开始，反复预测、保留高置信 token、重新 mask 低置信 token，最后得到完整文本。

所以 LLaDA 可以理解成:

```text
DDPM:   图像 x_0 -> 加高斯噪声 x_t -> 学会去噪
LLaDA: 文本 x_0 -> 随机 mask 成 x_t -> 学会补 mask
```

差别是，LLaDA 的噪声不是连续高斯噪声，而是离散 token 的 `[MASK]`。

## 论文主线

### 1. 作者先挑战 autoregressive 的“唯一性”

普通 LLM 采用 autoregressive modeling:

$$
p_\theta(x) = p_\theta(x^1)\prod_{i=2}^{L}p_\theta(x^i | x^1,\dots,x^{i-1})
$$

也就是:

```text
只能看左边上下文
一次预测一个 next token
生成顺序天然是从左到右
```

LLaDA 不否认 autoregressive 很成功，但它认为核心能力不一定来自“从左到右”本身，而可能来自更一般的 generative modeling:

$$
\max_\theta \mathbb{E}_{p_\text{data}(x)} \log p_\theta(x)
$$

换句话说，只要模型定义了一个合理的 $p_\theta(x)$，并且目标和 likelihood 有清楚关系，就可能 scale 出类似 LLM 的能力。

这就是 LLaDA 的切入点: 用 masked diffusion 定义语言模型分布。

### 2. Forward process: 按比例随机 mask token

给定干净文本序列:

```text
x_0 = [x_0^1, x_0^2, ..., x_0^L]
```

LLaDA 随机采样一个连续时间:

```text
t ~ Uniform(0, 1)
```

然后每个 token 独立地以概率 $t$ 被替换成 `[MASK]`:

$$
q_{t|0}(x_t|x_0)=\prod_{i=1}^{L}q_{t|0}(x_t^i|x_0^i)
$$

其中:

$$
q_{t|0}(x_t^i|x_0^i)=
\begin{cases}
1-t, & x_t^i = x_0^i \\
t, & x_t^i = \mathrm{M}
\end{cases}
$$

直觉非常简单:

| 时间 $t$ | 文本状态 |
|---:|---|
| $t=0$ | 完整文本，没有 mask |
| $t=0.3$ | 大约 30% token 被 mask |
| $t=0.8$ | 大约 80% token 被 mask |
| $t=1$ | 全部 token 都是 `[MASK]` |

这对应图像 diffusion 里的加噪过程:

```text
图像 diffusion: 干净图 -> 噪声越来越大 -> 纯噪声
LLaDA:          干净文本 -> mask 越来越多 -> 全 mask
```

### 3. Reverse process: 从全 mask 逐步补回文本

生成时方向反过来:

```text
r_1 = [MASK, MASK, MASK, ..., MASK]

for t = 1 -> 0:
    预测所有 masked positions 的 token
    保留一部分更确定的 token
    把另一部分重新 mask

得到 r_0
```

注意这里和 autoregressive 最大的区别是:

```text
AR:    每一步只生成最右边下一个 token
LLaDA: 每一步可以同时预测所有 masked token
```

所以 LLaDA 的生成顺序不是固定 left-to-right，而是由采样过程决定。一个后面的答案 token 可能先被预测出来，中间推理步骤可能后被稳定下来。

### 4. 核心网络是 mask predictor

LLaDA 用一个 Transformer 做 mask predictor:

$$
p_\theta(\cdot | x_t)
$$

输入是部分 masked 的序列 $x_t$，输出是每个 masked position 的 token 分布。

和 GPT 类模型相比，结构上的关键区别是:

| 模型 | Attention mask | 预测对象 |
|---|---|---|
| GPT / LLaMA | causal mask，只看左边 | next token |
| LLaDA | 不用 causal mask，可以看整个序列 | 所有 masked tokens |

LLaDA 仍然是 Transformer，不是不用 Transformer。它换的是 probabilistic modeling 方式，而不是抛弃 Transformer 架构。

一个重要细节是: LLaDA 的 Transformer 不需要输入 timestep embedding。论文引用 masked diffusion 的 time-free parameterization，认为如果只估计被 mask token 在干净数据中的条件分布，那么模型只需要看到当前哪些 token 是可见的、哪些是 `[MASK]`，不需要额外告诉它 $t$。

这和 DiT / DDPM 不一样:

```text
DDPM / DiT: x_t 的连续噪声强度需要 timestep t 告诉模型
LLaDA:      mask pattern 本身已经显式暴露了“噪声状态”
```

### 5. 训练目标: 只在 masked tokens 上算交叉熵

训练目标是:

$$
\mathcal{L}(\theta)
=
-\mathbb{E}_{t,x_0,x_t}
\left[
\frac{1}{t}
\sum_{i=1}^{L}
\mathbf{1}[x_t^i=\mathrm{M}]
\log p_\theta(x_0^i|x_t)
\right]
$$

它的意思是:

```text
随机选一段文本 x_0
随机选一个 mask ratio t
按 t 把 token mask 成 x_t
Transformer 看到 x_t
只对被 mask 的位置预测原 token
用 cross entropy 训练
```

伪代码:

```python
while training:
    x0 = sample_text()
    t = uniform(0, 1)

    xt = x0.copy()
    mask = bernoulli(p=t, shape=len(x0))
    xt[mask] = MASK

    logits = transformer_without_causal_mask(xt)
    loss = cross_entropy(logits[mask], x0[mask]) / t

    loss.backward()
    optimizer.step()
```

这里的 $\frac{1}{t}$ 不是随手加的。它来自 masked diffusion 的 likelihood bound。论文强调，这让 LLaDA 不只是“BERT 式填空”，而是一个有 likelihood lower/upper bound 关系的 generative model。

具体地，论文给出:

$$
-\mathbb{E}_{p_\text{data}(x_0)}[\log p_\theta(x_0)]
\le
\mathcal{L}(\theta)
$$

所以 LLaDA 的训练目标可以看作负对数似然的上界。

### 6. 和 BERT / MaskGIT 的区别

这篇论文很容易被误读成“大号 BERT”。但作者想强调的是:

| 模型 | mask ratio | 目标解释 | 生成方式 |
|---|---|---|---|
| BERT | 固定比例，经典是 15% | 表示学习 / MLM | 不是完整 generative model |
| MaskGIT | 多用于视觉 token，目标偏 heuristic | 并行 token refinement | 缺少同样的 likelihood 联系 |
| LLaDA | $t \sim U(0,1)$，覆盖所有 mask 比例 | masked diffusion / likelihood bound | 从全 mask 逐步反向生成 |

最关键的不是“能不能补 mask”，而是:

```text
LLaDA 把不同 mask ratio 下的补全任务统一成一个反向 diffusion process。
```

### 7. SFT: prompt 不动，只 mask response

监督微调时，数据是 prompt-response pair:

```text
(p_0, r_0)
```

LLaDA 不 mask prompt，只 mask response:

```text
input = [p_0, masked(r_0)]
target = r_0 中被 mask 的 token
```

目标是:

$$
-
\mathbb{E}_{t,p_0,r_0,r_t}
\left[
\frac{1}{t}
\sum_{i=1}^{L'}
\mathbf{1}[r_t^i=\mathrm{M}]
\log p_\theta(r_0^i|p_0,r_t)
\right]
$$

这很好理解:

```text
prompt 是条件，不能破坏
response 是要生成的对象，所以对 response 做 diffusion
```

它和 pre-training 完全兼容。可以把 `[prompt, response]` 看成一整段文本，只是 SFT 时所有 mask 都落在 response 部分。

### 8. Inference: 预测后再 remask

给定 prompt，LLaDA 先指定一个 response 长度 $L$，初始化为全 mask:

```text
r_1 = [MASK] * L
```

然后离散化 $t=1 \to 0$ 的反向过程。每一步做:

```python
for t in reversed_timesteps:
    s = t - step_size

    # 1. 预测所有 mask 位置
    logits = model(prompt, r_t)
    pred_tokens = argmax(logits)
    confidence = max_prob(logits)

    # 2. 已经 unmasked 的位置保持不变
    # 3. 在刚预测的 token 里，把低置信的一部分重新 mask
    #    remask 比例大致对应 s / t
    r_s = remask_low_confidence(pred_tokens, confidence, target_mask_ratio=s)
```

为什么要 remask？

因为如果每一步都把所有预测结果直接固定下来，就不是 diffusion 的逐步反向过程了。remasking 让模型可以反复修正低置信 token，类似图像 diffusion 中每一步只去掉一部分噪声。

论文默认使用 low-confidence remasking:

```text
置信度高的 token 先稳定下来
置信度低的 token 继续保持 mask，留给后续步骤再改
```

消融显示，low-confidence remasking 明显好于 random remasking。例如 LLaDA 8B Base 在若干生成任务上的对比:

| Remasking | BBH | GSM8K | Math | HumanEval | MBPP |
|---|---:|---:|---:|---:|---:|
| Random | 32.1 | 21.3 | 9.2 | 11.6 | 21.0 |
| Low-confidence | 45.0 | 70.0 | 30.3 | 32.9 | 40.2 |

### 9. 为什么它能缓解 reversal curse？

Autoregressive 模型天然按一个方向建模:

```text
x_1 -> x_2 -> x_3 -> ... -> x_L
```

如果训练里大量出现:

```text
A is B
```

它很容易学会从 A 推 B，但不一定自然学会从 B 推 A。这就是 reversal curse 的一种直觉来源。

LLaDA 的训练不是固定顺序的 next-token prediction，而是任意位置可能被 mask:

```text
[A] is [MASK]
[MASK] is B
[MASK] is [MASK]
A [MASK] B
```

因此它在训练中更像是在学习:

```text
任意可见 token 条件下，补任意不可见 token
```

这使它天然接近 any-order autoregressive 的味道，也解释了为什么论文中特别强调 reversal reasoning。

### 10. 主要实验结论

LLaDA 8B 从头预训练:

- 训练数据: 2.3T tokens。
- 序列长度: 4096。
- 计算量: 0.13 million H800 GPU hours。
- SFT: 4.5M pairs。
- 架构: LLaMA-like Transformer，RMSNorm、SwiGLU、RoPE。
- 注意力: 不使用 causal mask。

预训练模型和 LLaMA3 8B Base 的部分对比:

| Task | LLaDA 8B Base | LLaMA3 8B Base |
|---|---:|---:|
| MMLU | 65.9 | 65.4 |
| TruthfulQA | 46.1 | 44.0 |
| GSM8K | 70.3 | 48.7 |
| Math | 31.4 | 16.0 |
| HumanEval | 35.4 | 34.8 |
| HumanEval-FIM | 73.8 | 73.3 |
| CMMLU | 69.9 | 50.7 |
| C-Eval | 70.5 | 51.7 |

但它不是全面赢:

| Task | LLaDA 8B Base | LLaMA3 8B Base |
|---|---:|---:|
| BBH | 49.7 | 62.1 |
| ARC-C | 45.9 | 53.1 |
| Hellaswag | 70.5 | 79.1 |
| PIQA | 73.6 | 80.6 |
| MBPP | 40.0 | 48.8 |

比较稳妥的读法是:

```text
LLaDA 证明 diffusion language model 可以 scale 到 8B，
并且在不少 LLM benchmark 上接近同量级 AR 模型；
但它还不是一个全面替代 autoregressive LLM 的终局形态。
```

### 11. 采样策略: pure diffusion 整体最好

LLaDA 支持多种采样:

| 采样方式 | 直觉 |
|---|---|
| Autoregressive sampling | 强行按从左到右生成 |
| Block diffusion | 块之间自回归，块内 diffusion |
| Block Diffusion LLaDA | 固定块长度的半自回归 remasking |
| Pure diffusion | 整段 response 一起做 diffusion |

论文消融里，LLaDA 8B Base 上 pure diffusion 整体最好:

| Sampling | BBH | GSM8K | Math | HumanEval | MBPP |
|---|---:|---:|---:|---:|---:|
| Autoregressive | 38.1 | 63.1 | 23.6 | 18.3 | 33.4 |
| Block Diffusion LLaDA, $L'=32$ | 48.3 | 70.3 | 31.2 | 32.3 | 40.0 |
| Pure Diffusion | 49.7 | 70.3 | 31.4 | 35.4 | 40.0 |

这点很重要: LLaDA 不是“训练用 diffusion，推理还是最好 AR”。它的主结果确实来自 diffusion sampling。

### 12. 效率和局限

LLaDA 的并行生成看起来很诱人，但论文并没有宣称它全面更快。几个现实问题:

- LLaDA 目前不能像 causal LLM 那样自然使用 KV cache。
- 它通常需要固定一个 response length，然后从全 mask 开始采样。
- sampling steps 是质量和速度之间的旋钮。
- 对 SFT 后的模型，EOS padding 会影响 pure diffusion sampling，论文里还需要对 EOS confidence 做处理。
- 主结果没有使用 RL alignment，只用了 SFT。

论文在效率分析里说得比较克制: LLaDA 的目标不是立刻做出比 AR 更快的 LLM，而是证明 diffusion 语言模型在规模化后也能出现 LLM-like capabilities。

## 和 DDPM / DiT 的连接

如果把这篇放到 diffusion 脉络里，它的位置大概是:

```text
DDPM:
    连续图像空间
    加高斯噪声
    U-Net 预测噪声

DiT:
    latent image token
    加高斯噪声
    Transformer denoiser

LLaDA:
    离散文本 token
    随机 mask 噪声
    Transformer mask predictor
```

它和 DiT 都在讲“Transformer 可以做 diffusion backbone”，但重点不同:

| 论文 | 数据 | diffusion 噪声 | Transformer 的角色 |
|---|---|---|---|
| DiT | 图像 latent patch | Gaussian noise | denoiser，预测噪声/方差 |
| LLaDA | 文本 token | `[MASK]` corruption | mask predictor，预测原 token |

所以 LLaDA 最有意思的地方不是“又一个语言模型”，而是把 diffusion 的核心范式从连续视觉空间迁移到了离散语言空间:

```text
破坏数据 -> 学会反向恢复 -> 从最大破坏状态开始生成
```

## 需要记住的点

1. LLaDA = Large Language Diffusion with mAsking。
2. 它不是自回归 next-token prediction，而是 masked diffusion。
3. Forward process 是按 $t$ 随机 mask token，$t=1$ 时全 mask。
4. Reverse process 是从全 mask response 开始，反复预测和 remask。
5. 训练 loss 只在 masked tokens 上算 CE，并有 $\frac{1}{t}$ 权重。
6. 这个 loss 是负 log-likelihood 的上界，因此比普通 MLM 更像完整 generative modeling。
7. Transformer 不用 causal mask，也不需要显式 timestep embedding。
8. SFT 时 prompt 不 mask，只对 response 做 masked diffusion。
9. Pure diffusion sampling 在论文消融里整体最好。
10. 它证明 diffusion LLM 有希望，但效率、KV cache、长度控制、alignment 仍然是开放问题。

## 我的理解

LLaDA 的价值不在于马上替代 GPT，而在于它把一个长期默认前提拆开了:

```text
LLM 能力 = Transformer + 大数据 + 大模型 + 生成式目标
```

这里面 “生成式目标” 不一定只能是:

```text
从左到右预测下一个 token
```

也可以是:

```text
任意位置被遮住后，学会把整段文本恢复出来
```

如果 DDPM 让我们相信“从噪声里逐步恢复图像”可以成为强生成模型，那么 LLaDA 想让我们相信:

```text
从全 mask 里逐步恢复文本，也可能成为强语言模型。
```

这也是它和当前 diffusion 组笔记的连接点。
