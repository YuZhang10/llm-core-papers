---
title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
short_title: "ViT"
paper_id: "arXiv:2010.11929"
authors: "Alexey Dosovitskiy et al."
venue: "ICLR 2021"
date: "2026-05-15"
tags:
  - vision-transformer
  - transformer
  - image-recognition
  - patch-token
  - scaling
---

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

## 本地文件

- PDF: [ViT_An_Image_is_Worth_16x16_Words_arXiv_2010.11929.pdf](ViT_An_Image_is_Worth_16x16_Words_arXiv_2010.11929.pdf)
- arXiv 源码包: [ViT_arXiv_2010.11929_source.tar.gz](ViT_arXiv_2010.11929_source.tar.gz)
- arXiv 源码目录: [ViT_arXiv_2010.11929_source](ViT_arXiv_2010.11929_source)
- arXiv: https://arxiv.org/abs/2010.11929

## 一句话理解

ViT 的核心问题是: **图像识别能不能像 NLP 一样，直接把输入变成 token 序列，然后交给标准 Transformer encoder？**

答案是可以，但有一个前提: **数据规模要足够大**。在小数据上，缺少 CNN 的局部性和二维结构归纳偏置会吃亏；在 JFT-300M / ImageNet-21k 这种大规模预训练后，纯 Transformer 可以超过强 CNN baseline。

## 论文主线

### 1. Transformer 在 NLP 成功，但视觉里仍主要是 CNN

在 ViT 之前，视觉模型可以用 attention，但通常还是带着 CNN 主干或视觉专用 attention 结构:

- CNN + attention 模块。
- 局部 attention。
- axial attention。
- hybrid CNN-Transformer。

作者选择了一条更“硬”的路线: 尽量少引入图像专用设计，把图像切成 patch，当成 token 序列，直接送进标准 Transformer encoder。

### 2. 把图像切成 patch token

输入图像:

$$
x \in \mathbb{R}^{H \times W \times C}
$$

切成固定大小 patch:

$$
x_p \in \mathbb{R}^{N \times (P^2 C)}
$$

其中:

$$
N = HW / P^2
$$

这里:

- $H,W$: 图像高宽。
- $C$: 通道数，比如 RGB 图像 $C=3$。
- $P$: patch 的边长，例如 16。
- $N$: patch 数，也就是 Transformer 的有效序列长度。

例如一张 224x224x3 图像，patch size 是 16:

```text
224 / 16 = 14
patch 数 = 14 x 14 = 196
每个 patch 原始维度 = 16 x 16 x 3 = 768
```

然后每个 patch 被 flatten 成向量，再通过线性层投影到 hidden size $D$:

$$
\mathbb{R}^{P^2 C} \to \mathbb{R}^{D}
$$

这一步得到 patch embedding。

## ViT 架构

ViT 的结构非常接近 BERT:

```text
image
  -> split into patches
  -> flatten each patch
  -> linear projection to D-dim patch embeddings
  -> prepend [CLS] token
  -> add position embeddings
  -> Transformer encoder layers
  -> use final [CLS] representation for classification
```

### Patch embedding

论文中的输入写法:

$$
z_0 = [x_\text{class}; x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{pos}
$$

其中:

- $x_\text{class}$ 是可学习的 classification token。
- $E$ 是 patch projection 矩阵。
- $E_{pos}$ 是位置 embedding。

### Transformer encoder block

每层基本是标准 pre-LN Transformer:

$$
z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}
$$

$$
z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell
$$

最后用 class token 的输出作为图像表示:

$$
y = \mathrm{LN}(z_L^0)
$$

### 位置 embedding

ViT 使用 learnable 1D position embedding。虽然输入 patch 来自二维图像网格，但作者发现更复杂的 2D-aware positional embedding 没带来显著收益。

这点很重要: ViT 对二维结构的手工假设非常少，主要只在两个地方显式用到图像结构:

1. 最开始把图像切成 patch。
2. fine-tune 到更高分辨率时，对 position embedding 做 2D 插值。

## 为什么 patch size 很关键？

patch size 控制序列长度。

如果图像是 224x224:

| Patch size | Patch 网格 | Token 数，不含 CLS | 直觉 |
|---|---:|---:|---|
| 32 | 7 x 7 | 49 | 便宜但粗 |
| 16 | 14 x 14 | 196 | 常用平衡点 |
| 14 | 16 x 16 | 256 | 更细但更贵 |

self-attention 的主要计算和 token 数大致呈二次关系，所以 patch 越小，空间细节越多，计算也越贵。

这和 DiT 里的 latent patch size 是同一类思想:

```text
ViT: 原图 patch -> classification token sequence
DiT: latent patch -> denoising token sequence
```

## 模型规模

ViT 的模型大小基本按 BERT 配置来:

| Model | Layers | Hidden size D | MLP size | Heads | Params |
|---|---:|---:|---:|---:|---:|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

命名方式:

- ViT-B/16: Base 模型，16x16 patch。
- ViT-L/16: Large 模型，16x16 patch。
- ViT-H/14: Huge 模型，14x14 patch。

## 实验设置

作者重点验证的是大规模预训练后的迁移能力。

预训练数据:

- ImageNet-1k: 1.3M images, 1k classes。
- ImageNet-21k: 14M images, 21k classes。
- JFT-300M: 303M images, 18k classes。

训练:

- 预训练用 Adam。
- batch size 4096。
- weight decay 0.1。
- 线性 warmup 和 decay。
- fine-tuning 用 SGD momentum。
- ImageNet final result 会在更高分辨率上 fine-tune，比如 ViT-L/16 用 512，ViT-H/14 用 518。

## 关键实验结果

### 大数据预训练后，ViT 能超过强 CNN

论文 Table 2 中，JFT-300M 预训练后的结果:

| Model | ImageNet | ImageNet ReaL | CIFAR-10 | CIFAR-100 | VTAB |
|---|---:|---:|---:|---:|---:|
| ViT-H/14, JFT | 88.55 | 90.72 | 99.50 | 94.55 | 77.63 |
| ViT-L/16, JFT | 87.76 | 90.54 | 99.42 | 93.90 | 76.28 |
| ViT-L/16, ImageNet-21k | 85.30 | 88.62 | 99.15 | 93.25 | 72.72 |
| BiT-L, ResNet152x4 | 87.54 | 90.54 | 99.37 | 93.51 | 76.29 |

ViT-H/14 在多项 benchmark 上超过了 BiT-L，并且预训练 TPUv3-core-days 更少。

### ViT 的数据需求高

论文最有价值的结论之一不是“Transformer 永远更好”，而是:

> 缺少 CNN 归纳偏置的 ViT，在小数据上表现不如强 CNN；但当预训练数据足够大时，scale 会反过来成为优势。

作者做了 ImageNet、ImageNet-21k、JFT-300M 的对比:

- 只用 ImageNet-1k 预训练，ViT 往往不如 ResNet/BiT。
- 用 ImageNet-21k，差距收窄。
- 用 JFT-300M，大模型优势显现，ViT-L/ViT-H 超过 CNN。

这和后来大模型路线非常一致: 弱归纳偏置 + 大数据 + 大模型，换取更强的 scaling。

### self-attention 确实学到全局和局部混合

论文分析 attention distance:

- 一些低层 attention head 已经能看很远。
- 也有一些低层 head 保持局部关注，类似 CNN 早期局部特征。
- 随着层数变深，attention distance 增大，更多 head 关注全局区域。

所以 ViT 不是完全抛弃图像局部性，而是让模型自己从数据中学出来。

## 和 CNN 的核心差异

| 维度 | CNN | ViT |
|---|---|---|
| 输入形式 | 2D feature map | patch token sequence |
| 基本操作 | convolution | self-attention + MLP |
| 局部性 | 强手工归纳偏置 | 需要从数据中学 |
| 平移等变性 | 天然较强 | 主要靠数据和位置编码 |
| 长距离关系 | 需要堆层/扩大感受野 | 每层 attention 可全局交互 |
| 数据需求 | 相对低 | 高 |
| scaling 潜力 | 强但结构更视觉专用 | 非常适合借用 NLP Transformer 生态 |

## 与 DiT 的连接

DiT 几乎直接继承了 ViT 的 token 化方式:

```text
ViT:
image 224x224x3
  -> 16x16 image patches
  -> patch embeddings
  -> Transformer encoder
  -> classification

DiT:
latent 32x32x4
  -> 2x2 latent patches
  -> patch embeddings
  -> Transformer blocks with timestep/class conditioning
  -> noise/covariance prediction
```

两者最大区别:

- ViT 是 discriminative model，输出类别。
- DiT 是 generative diffusion denoiser，输出和输入 latent 同形状的噪声预测。

但“把二维空间切成 patch token，再用 Transformer 处理”的基础动作完全同源。

## 局限性

1. 数据需求高: ViT 在小数据上不一定赢 CNN。
2. patch 化会丢掉 patch 内更细的局部建模，需要靠 projection 和后续层学习。
3. attention 成本随 token 数增加很快，高分辨率视觉任务会很贵。
4. 原始 ViT 主要是分类模型，后来的检测/分割/生成需要额外设计。

## 读这篇时抓住什么

1. 图像如何变成 token: patch -> flatten -> linear projection。
2. class token 如何作为整张图表示。
3. position embedding 为什么必要。
4. ViT 为什么小数据不占优，大数据预训练后占优。
5. patch size、sequence length、compute 三者的关系。

## 我的评价

综合评分: 9.5/10。

这篇论文的重要性不在于模块复杂，而在于它把视觉模型的默认架构从 CNN 推向了 Transformer。后来的 MAE、Swin、DINO、CLIP vision encoder、DiT 等很多路线都可以从 ViT 这里接上。

最值得记住的三句话:

1. ViT 把图像切成 patch，把 patch 当成 token。
2. ViT 减少了图像专用归纳偏置，因此更依赖大规模预训练。
3. 当数据和模型足够大时，标准 Transformer 在视觉上也能 scale。

