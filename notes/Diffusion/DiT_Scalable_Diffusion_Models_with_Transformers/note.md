---
title: "Scalable Diffusion Models with Transformers"
short_title: "DiT"
paper_id: "arXiv:2212.09748"
authors: "William Peebles, Saining Xie"
venue: "ICCV 2023 Oral"
date: "2026-05-14"
tags:
  - diffusion
  - transformer
  - DiT
  - latent-diffusion
  - image-generation
---

# Scalable Diffusion Models with Transformers

## 本地文件

- PDF: [Scalable_Diffusion_Models_with_Transformers_DiT_arXiv_2212.09748.pdf](Scalable_Diffusion_Models_with_Transformers_DiT_arXiv_2212.09748.pdf)
- arXiv 源码包: [DiT_arXiv_2212.09748_source.tar.gz](DiT_arXiv_2212.09748_source.tar.gz)
- arXiv 源码目录: [DiT_arXiv_2212.09748_source](DiT_arXiv_2212.09748_source)
- 论文主页: https://www.wpeebles.com/DiT
- arXiv: https://arxiv.org/abs/2212.09748

## 一句话理解

DiT 的核心不是发明新的 diffusion objective，而是把 latent diffusion 里的 U-Net denoiser 换成一个 ViT 风格 Transformer，并证明在图像扩散模型里，Transformer 也有很清楚的 scaling behavior: 模型 forward Gflops 越高，FID 越低。

换句话说，这篇论文回答的问题是:

> 扩散模型真的需要 U-Net 吗？还是可以像 NLP 和视觉识别一样，直接用 Transformer，然后靠规模化获得更好的生成质量？

论文给出的答案是: 可以。关键做法是 latent patch tokens + adaLN-Zero 条件注入 + 按 Gflops 扩展模型。

## 论文主线

### 1. 以前的图像 diffusion 基本都用 U-Net

DDPM、ADM、LDM、Imagen、GLIDE 等图像扩散模型，主干网络大多是卷积 U-Net。U-Net 的优势很自然:

- 卷积带来局部归纳偏置。
- 多尺度 downsample/upsample 适合图像。
- skip connection 保留空间细节。
- diffusion 每一步都像图像到图像的去噪任务，U-Net 很顺手。

但作者认为这里有一个值得拆开的假设: U-Net 是不是扩散模型成功的必要条件？

Transformer 在语言、视觉分类、视觉生成等任务上已经显示出强 scaling 能力。如果 diffusion 的 denoiser 也能换成 Transformer，那么扩散模型也可以吃到 Transformer 生态里的 scaling recipe。

### 2. 直接在像素上做 Transformer 太贵，所以站在 LDM 上

DiT 没有直接在 RGB 像素空间生成，而是沿用 Latent Diffusion Models 的框架:

1. 用预训练 VAE encoder 把图像 $x$ 压缩成 latent $z = E(x)$。
2. 在 latent space 里训练 diffusion model。
3. 采样完成后，用 VAE decoder 把 latent 还原成图像 $x = D(z)$。

对 256x256 RGB 图像，Stable Diffusion 风格的 VAE 会把它压成约 $32 \times 32 \times 4$ 的 latent。这样 Transformer 处理的是一个小得多的空间表示，而不是原始像素。

这一点很重要: DiT 不是“纯 Transformer 端到端像素生成”，而是“卷积 VAE + Transformer diffusion denoiser”的混合路线。

### 3. 把 latent feature map 切成 patch token

DiT 的输入是某个时间步的 noisy latent $z_t$，形状是 $I \times I \times C$。例如 256x256 图像对应 $I=32, C=4$。

然后像 ViT 一样 patchify:

$$
T = (I / p)^2
$$

其中 $p$ 是 latent patch size，$T$ 是 token 数。

对 256x256 图像的 $32 \times 32 \times 4$ latent:

| Patch size | Token 数 | 直觉 |
|---|---:|---|
| $p=8$ | 16 | token 少，计算便宜，细节能力弱 |
| $p=4$ | 64 | 中等 |
| $p=2$ | 256 | token 多，计算贵，生成质量更好 |

对 512x512 图像，latent 是 $64 \times 64 \times 4$。如果仍然用 $p=2$，token 数变成 1024。

这也是论文最重要的 scaling 变量之一: patch 越小，token 越多，self-attention 计算越大，FID 通常越好。

### 4. DiT block 基本是 ViT block，但 diffusion 需要条件注入

普通 ViT block 处理图像 token，但 diffusion denoiser 还必须知道:

- 当前 timestep $t$，因为不同噪声强度下去噪任务不同。
- class label $c$，因为论文做的是 ImageNet class-conditional generation。

作者比较了四种把 $t$ 和 $c$ 注入 Transformer 的方式:

| 条件注入方式 | 做法 | 结果 |
|---|---|---|
| In-context conditioning | 把 timestep/class embedding 当成额外 token 拼到序列里 | 简单，但效果差 |
| Cross-attention | 图像 token 对条件 token 做 cross-attention | 有效，但多约 15% Gflops |
| adaLN | 用条件 embedding 生成 LayerNorm 的 scale/shift | 计算开销小，效果好 |
| adaLN-Zero | adaLN 之外再加 residual gate，并零初始化残差分支 | 最好，后续所有模型采用 |

### 5. adaLN-Zero 是架构里的关键小改动

adaLN 的基本思想是: LayerNorm 不再使用固定的 $\gamma, \beta$，而是从 timestep embedding 和 class embedding 的和中回归出每层的调制参数。

adaLN-Zero 再进一步:

- 除了回归 LayerNorm 的 scale/shift，还回归残差分支前的缩放参数 $\alpha$。
- 初始化时让 $\alpha=0$。
- 这样每个 DiT block 一开始近似 identity function。

直觉上，扩散模型每个 block 初始时先“不乱动”，训练中再逐步学会如何修改表示。这个想法和 ResNet/Diffusion U-Net 里零初始化残差分支的经验一致。

论文里的消融非常清楚。以 DiT-XL/2、400K steps、无 classifier-free guidance 为例:

| Block design | Gflops | Params | FID-50K |
|---|---:|---:|---:|
| In-context | 119.37 | 449M | 35.24 |
| Cross-attention | 137.62 | 598M | 26.14 |
| adaLN | 118.56 | 600M | 25.21 |
| adaLN-Zero | 118.64 | 675M | 19.47 |

所以 DiT 不是“随便把 U-Net 换成 Transformer 就好了”。真正让它稳定、高质量工作的条件注入方式是 adaLN-Zero。

## DiT 的完整前向流程

可以把 DiT 的 forward pass 理解成:

```text
image x
  -> VAE encoder E
latent z
  -> add diffusion noise at timestep t
noisy latent z_t
  -> patchify into latent tokens
tokens + positional embeddings
  -> N DiT blocks with timestep/class conditioning via adaLN-Zero
decoded tokens
  -> unpatchify
predicted noise and covariance
  -> diffusion loss / reverse sampling
  -> VAE decoder D
generated image
```

输出不是直接图像，而是 diffusion 所需的预测量:

- 预测噪声 $\epsilon_\theta(z_t, t, c)$。
- 预测反向过程的 diagonal covariance。

最后线性层会把每个 token 解码成 $p \times p \times 2C$，再 unpatchify 回原 latent 的空间形状。

## 模型规模设计

DiT 沿用 ViT 的 S/B/L 风格，并额外加入 XL:

| Model | Layers | Hidden size | Heads | Gflops, $I=32,p=4$ |
|---|---:|---:|---:|---:|
| DiT-S | 12 | 384 | 6 | 1.4 |
| DiT-B | 12 | 768 | 12 | 5.6 |
| DiT-L | 24 | 1024 | 16 | 19.7 |
| DiT-XL | 28 | 1152 | 16 | 29.1 |

命名方式是 `模型大小/patch size`。例如:

- DiT-XL/2: XL 规模，latent patch size 为 2。
- DiT-B/4: Base 规模，latent patch size 为 4。

注意 `/2` 不是 diffusion step，也不是图像分辨率，而是 latent patch size。

## 实验设置

主要实验是在 ImageNet class-conditional generation 上做:

- 分辨率: 256x256 和 512x512。
- Diffusion space: Stable Diffusion/LDM 风格 VAE latent。
- VAE downsample factor: 8。
- 训练优化器: AdamW。
- 学习率: $1 \times 10^{-4}$。
- Batch size: 256。
- Weight decay: 0。
- 数据增强: horizontal flip。
- EMA: 0.9999。
- Diffusion schedule: ADM 的 1000-step linear variance schedule。
- 评估: FID-50K 为主，另有 sFID、Inception Score、Precision/Recall。
- 采样: 和 prior work 对比时使用 250 DDPM sampling steps。
- 实现: JAX，TPU-v3 pods。

作者特别强调: 这些超参基本沿用 ADM，没有为不同 DiT 大小单独调学习率、warmup、weight decay 等。

## 最重要的实验发现

### 发现 1: Gflops 比参数量更能解释生成质量

论文最强的结论是: DiT 的 FID 和 forward-pass Gflops 强相关。

12 个模型在 256x256 ImageNet、400K steps、无 guidance 下的结果:

| Model | Gflops | Params | FID-50K |
|---|---:|---:|---:|
| DiT-S/8 | 0.36 | 33M | 153.60 |
| DiT-S/4 | 1.41 | 33M | 100.41 |
| DiT-S/2 | 6.06 | 33M | 68.40 |
| DiT-B/8 | 1.42 | 131M | 122.74 |
| DiT-B/4 | 5.56 | 130M | 68.38 |
| DiT-B/2 | 23.01 | 130M | 43.47 |
| DiT-L/8 | 5.01 | 459M | 118.87 |
| DiT-L/4 | 19.70 | 458M | 45.64 |
| DiT-L/2 | 80.71 | 458M | 23.33 |
| DiT-XL/8 | 7.39 | 676M | 106.41 |
| DiT-XL/4 | 29.05 | 675M | 43.01 |
| DiT-XL/2 | 118.64 | 675M | 19.47 |

这里有两个很有意思的对照:

- DiT-S/2 和 DiT-B/4 参数量差很多，但 Gflops 接近，FID 几乎一样: 68.40 vs 68.38。
- DiT-XL/8 有 676M 参数，但 token 太少、Gflops 只有 7.39，FID 反而比小很多的 DiT-S/2 差。

所以这篇论文想让读者记住的是: 在 DiT 里，参数量不是充分解释变量，token 数和 forward compute 更关键。

### 发现 2: 同时扩大模型大小和 token 数都有效

作者扩了两个轴:

1. 模型大小: S -> B -> L -> XL。
2. token 数: patch size 8 -> 4 -> 2。

结论是两个方向都有效:

- 固定 patch size，模型更深更宽，FID 更好。
- 固定模型大小，patch size 更小、token 更多，FID 更好。

这和 Transformer 的常见 scaling 直觉一致: 模型容量和序列长度处理能力都在提升表达能力。

### 发现 3: 大模型在训练 compute 上也更划算

作者不仅看固定 step 的 FID，还看达到某个 FID 所需的总训练 compute。

总训练 compute 估算为:

$$
\text{training compute} = \text{model Gflops} \times \text{batch size} \times \text{training steps} \times 3
$$

其中 $\times 3$ 粗略表示 forward + backward 的开销。

结果是: 小模型训练再久也会逐渐变得不划算，大模型在高 compute 区间更有效率。这个结论很像语言模型 scaling: 小模型可以低成本起步，但一旦预算足够，直接训练更大的模型更好。

### 发现 4: 多采样步数不能弥补小模型本身的不足

Diffusion 有一个特殊点: 推理时可以增加 sampling steps，多花 test-time compute。

作者专门问: 小模型多采样几步，能不能追上大模型？

答案基本是否定的。论文中的例子:

- DiT-L/2 用 1000 sampling steps，每张图约 80.7 Tflops，FID-10K 为 25.9。
- DiT-XL/2 用 128 sampling steps，每张图约 15.2 Tflops，FID-10K 为 23.7。

也就是说，小模型推理时花更多计算，也不一定能补上 model compute 的差距。

### 发现 5: DiT-XL/2 在 ImageNet 上达到当时 SOTA

256x256 ImageNet:

| Model | FID | IS | Precision | Recall |
|---|---:|---:|---:|---:|
| LDM-4-G, cfg=1.50 | 3.60 | 247.67 | 0.87 | 0.48 |
| StyleGAN-XL | 2.30 | 265.12 | 0.78 | 0.53 |
| DiT-XL/2-G, cfg=1.50 | 2.27 | 278.24 | 0.83 | 0.57 |

512x512 ImageNet:

| Model | FID | IS | Precision | Recall |
|---|---:|---:|---:|---:|
| ADM-G + ADM-U | 3.85 | 221.72 | 0.84 | 0.53 |
| DiT-XL/2-G, cfg=1.50 | 3.04 | 240.82 | 0.84 | 0.54 |

无 guidance 时，DiT-XL/2 的 FID 没有 guidance 版本那么夸张:

- 256x256: 9.62。
- 512x512: 12.03。

但加 classifier-free guidance 后，FID 大幅下降。这一点和其他 conditional diffusion 模型一致。

## 我自己的串联理解

这篇文章可以看成三层故事。

第一层是架构替换: 原来 diffusion denoiser 默认是 U-Net，现在换成 Transformer。这个替换本身听起来直接，但真正的难点是 diffusion 不只是图像 token 建模，它还强依赖 timestep 和条件信息。因此 DiT 需要一个适合 Transformer 的条件注入机制。adaLN-Zero 是让这个替换成立的关键工程点。

第二层是 latent token 化: 如果直接在像素上使用 Transformer，序列长度会爆炸。DiT 借助 LDM 的 VAE latent，把图像变成较小的空间特征图，再 patchify 成 token。这样它既保留了图像的二维空间结构，又能使用 ViT 式 Transformer。

第三层是 scaling 论证: 作者真正想证明的不是“我有一个新结构比 U-Net 好一点”，而是“Transformer 作为 diffusion backbone 也有可预测的规模化曲线”。因此论文用 Gflops 而不是参数量作为主轴，展示更高模型计算量和更好 FID 之间的关系。

如果把它放到后来的生成模型发展里看，DiT 的意义更大。它打开了 diffusion backbone 从 U-Net 迁移到 Transformer 的路线。后来的很多图像和视频生成模型，包括一批 text-to-image / text-to-video diffusion transformer，都可以看作沿着这条路线继续放大模型、token、条件模态和数据规模。

## 和 U-Net diffusion 的关键差异

| 维度 | U-Net diffusion | DiT |
|---|---|---|
| 主干 | 卷积 ResNet block + attention | ViT/Transformer block |
| 空间结构 | 多尺度 encoder-decoder | patch token 序列 |
| 条件注入 | adaptive normalization 很常见 | adaLN-Zero |
| scaling 轴 | channel 数、resolution、多尺度结构 | depth/width、token 数、Gflops |
| 归纳偏置 | 强图像局部性、多尺度 | 弱局部偏置，更通用 |
| 优势 | 图像任务成熟、计算友好 | 更适合规模化和跨模态统一 |
| 风险 | 架构更特化 | attention 对 token 数敏感，成本高 |

## 这篇论文的贡献

### 贡献 1: 提出 Diffusion Transformer 作为 diffusion backbone

DiT 证明了扩散模型不必绑定 U-Net。只要设计好 latent token 化和条件注入，Transformer 可以直接作为 denoiser。

### 贡献 2: 找到有效的条件注入方式 adaLN-Zero

作者系统比较了 in-context、cross-attention、adaLN、adaLN-Zero，发现 adaLN-Zero 在质量和计算效率上最好。

### 贡献 3: 给出 DiT 的 scaling 规律

论文用 12 个模型说明，FID 和 forward Gflops 强相关。模型变大、token 变多都会改善生成质量。

### 贡献 4: 在 ImageNet class-conditional 生成上达到强结果

DiT-XL/2 在 256x256 和 512x512 ImageNet 上都超过了 prior diffusion baseline，256x256 达到 FID 2.27。

## 局限性

### 1. 任务仍然是 ImageNet class-conditional

论文不是 text-to-image，也没有处理复杂文本条件。它验证的是 class label 条件下的图像生成，和实际开放域文生图还有距离。

### 2. 仍然依赖卷积 VAE

DiT 替换的是 diffusion denoiser，不是整个生成系统。VAE encoder/decoder 仍然是卷积模型，latent quality 也会影响最终图像。

### 3. Transformer attention 成本随 token 数上升很快

patch size 从 4 到 2，token 数变 4 倍，attention 成本明显上升。512x512 下 DiT-XL/2 已经有 1024 tokens 和 524.6 Gflops。

### 4. 主要评价仍围绕 FID

论文也报告 IS、sFID、Precision/Recall，但主线强依赖 FID。FID 对评估实现细节敏感，也不能完整代表人类偏好、多样性和语义一致性。

### 5. 计算门槛高

最强模型使用 TPU-v3 pod 训练，DiT-XL/2 256x256 训练到 7M steps，512x512 训练到 3M steps。复现实验成本不低。

## 阅读这篇论文时应该盯住什么

如果时间有限，建议按这个顺序读:

1. Introduction: 看作者为什么要挑战 U-Net。
2. Section 3 Diffusion Transformers: 重点看 patchify、condition block、adaLN-Zero。
3. Table 1: 理解 S/B/L/XL 的规模配置。
4. Figure 4/5/6 和 Appendix Table 3: 理解 Gflops-FID scaling。
5. Table 2: 看最终 ImageNet SOTA 结果。
6. Section 4.2 Sampling Compute: 看为什么多采样步数不能替代模型本身的 compute。

## 常见疑问

### DiT 和 ViT 的关系是什么？

DiT 很像 ViT，但任务不同。ViT 做图像分类时通常输出 class token 或 pooled representation；DiT 做 diffusion denoising，需要输出和输入 latent 同形状的噪声/方差预测。因此 DiT 有 patchify、Transformer blocks、linear decoder、unpatchify 这一套。

### DiT-XL/2 里的 `/2` 是什么意思？

`/2` 是 latent patch size $p=2$。对 256x256 图像，latent 是 $32 \times 32$，所以 token 数是 $(32/2)^2=256$。

### 为什么不是参数越多越好？

因为图像生成质量强烈依赖处理多少 spatial token。DiT-XL/8 参数很多，但 token 太少，Gflops 不高，FID 很差。DiT-S/2 参数少得多，但 token 多、Gflops 接近，FID 反而更好。

### adaLN-Zero 为什么有效？

它一方面把 timestep/class 信息注入每层归一化，另一方面让每个残差分支初始近似 0，使 block 一开始接近 identity。这样训练更稳，条件信息也更直接地影响 denoising。

### 这篇论文和 Stable Diffusion 的关系是什么？

DiT 借用了 Stable Diffusion/LDM 的 VAE latent space 思路，但把 latent diffusion 里的 U-Net denoiser 换成 Transformer。它不是 Stable Diffusion 的完整替代，因为本文主要是 ImageNet class-conditional，不是文本条件开放域生成。

## 后续影响

这篇论文的长期影响是把 diffusion model 的 backbone 从“U-Net 是默认选择”推向“Transformer 是可扩展主干”。这条路线后来影响了很多大规模图像和视频生成模型。

可以把发展线粗略理解为:

```text
DDPM / ADM: pixel-space U-Net diffusion
  -> LDM / Stable Diffusion: latent-space U-Net diffusion
  -> DiT: latent-space Transformer diffusion
  -> 后续大规模 text-to-image / video diffusion transformers
```

## 我的评价

综合评分: 9/10。

它的创新不是某个复杂 trick，而是把领域的默认假设拆掉，并用一组干净的 scaling 实验证明新路线可行。这类论文的价值通常不在于“模块多复杂”，而在于它改变后续研究者默认会尝试什么。

最值得记住的三句话:

1. DiT 把 latent diffusion 的 U-Net denoiser 换成 ViT-style Transformer。
2. adaLN-Zero 是让 timestep/class conditioning 在 Transformer denoiser 里工作得很好的关键设计。
3. DiT 的生成质量与 forward Gflops 强相关，token 数和模型规模一起构成主要 scaling 轴。

