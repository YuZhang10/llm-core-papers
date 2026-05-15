---
title: "Denoising Diffusion Probabilistic Models"
short_title: "DDPM"
paper_id: "arXiv:2006.11239"
authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel"
venue: "NeurIPS 2020"
date: "2026-05-15"
tags:
  - diffusion
  - ddpm
  - generative-model
  - denoising
  - score-matching
---

# Denoising Diffusion Probabilistic Models

## 本地文件

- PDF: [DDPM_Denoising_Diffusion_Probabilistic_Models_arXiv_2006.11239.pdf](DDPM_Denoising_Diffusion_Probabilistic_Models_arXiv_2006.11239.pdf)
- arXiv 源码包: [DDPM_arXiv_2006.11239_source.tar.gz](DDPM_arXiv_2006.11239_source.tar.gz)
- arXiv 源码目录: [DDPM_arXiv_2006.11239_source](DDPM_arXiv_2006.11239_source)
- arXiv: https://arxiv.org/abs/2006.11239

## 一句话理解

DDPM 把图像生成改写成一个很朴素的学习问题:

> 我给一张图加不同程度的高斯噪声，让神经网络学会“这张 noisy 图里混进了什么噪声”。生成时从纯噪声开始，反复把模型预测的噪声拿掉一点，最后得到图像。

这篇论文真正厉害的地方，是把看起来复杂的 diffusion / variational inference / score matching，落成了一个特别容易工程实现的训练目标: **随机时间步的噪声预测 MSE**。

## 论文主线

### 1. 正向过程: 人为固定的逐步加噪

DDPM 先定义一个不需要学习的 forward process:

```text
x_0 -> x_1 -> x_2 -> ... -> x_T
干净图像 -> 一点噪声 -> 更多噪声 -> 纯噪声
```

每一步做的事情就是:

```text
保留一点上一时刻的图像
再混入一点新的高斯噪声
```

最核心的加噪式子是:

$$
x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t
$$

其中 $\epsilon_t$ 是每一步重新采样的标准高斯噪声。

这里有一个容易忽略但很重要的设计: 旧信号前面乘了 $\sqrt{1-\beta_t}$。这不是装饰，而是为了让整体数值尺度稳定。模型应该学习“信号和噪声比例如何变化”，而不是被迫适应输入幅度一路变大。

### 2. 训练时不需要真的一步步加到 t

虽然概念上是一步步加噪，但高斯有一个好性质: 多次加高斯噪声，最后仍然等价于一次加高斯噪声。

所以可以直接从干净图 $x_0$ 跳到任意时间步 $x_t$:

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
$$

这句话可以当成 DDPM 最关键的直觉公式:

```text
x_t = 某个比例的干净图 + 某个比例的随机噪声
```

当 `t` 小，干净图比例高，噪声比例低。  
当 `t` 大，干净图比例低，噪声比例高。  
当 `t = T`，基本就是纯噪声。

这让训练变得很简单: 随机抽一个 `t`，直接构造对应噪声强度的训练样本。

### 3. 反向过程: 学会一步步去噪

生成时方向反过来:

```text
x_T -> x_{T-1} -> ... -> x_0
纯噪声 -> 稍微像图 -> 更像图 -> 清晰图像
```

DDPM 把每一步反向去噪建模成一个高斯转移。你可以先不管完整概率形式，只抓住直觉:

```text
当前 x_t
  -> U-Net 预测里面的噪声 epsilon_theta
  -> 根据预测噪声算出更干净一点的 x_{t-1}
  -> 重复很多次
```

它不是一步生成整张图，而是把生成拆成许多很小、很容易学的“去噪小步”。

### 4. 为什么预测噪声，而不是直接预测干净图？

这篇论文试过不同参数化。最后最有效的是让网络预测噪声:

```text
input: noisy image x_t, timestep t
output: predicted noise epsilon_theta
target: true noise epsilon
loss: MSE(predicted noise, true noise)
```

训练目标可以记成:

$$
\text{loss} = \|\epsilon - \epsilon_\theta(x_t,t)\|^2
$$

这个选择有几个好处:

- 目标简单，训练稳定。
- 网络只需要识别“哪些成分是噪声”。
- 它和 denoising score matching / Langevin dynamics 有理论联系。
- 实验上比直接预测均值配合简单 MSE 效果好得多。

所以现代 diffusion 里经常说的 “epsilon prediction”，源头就在这里。

## 训练流程

可以用很短的伪代码理解:

```text
repeat:
    取一张真实图 x_0
    随机抽一个时间步 t
    随机抽一份高斯噪声 epsilon
    按 t 的噪声比例混合出 x_t
    让 U-Net 看到 x_t 和 t
    U-Net 预测 epsilon_theta
    用 MSE 让 epsilon_theta 接近真实 epsilon
```

模型学到的不是“从零画图”，而是各种噪声强度下的“噪声识别能力”。

## 采样流程

生成时:

```text
从 x_T ~ N(0, I) 开始
for t = T, T-1, ..., 1:
    用 U-Net 预测当前 x_t 里的噪声
    去掉一点预测噪声
    加入一点受控随机性
得到 x_0
```

论文里通常用 $T=1000$，所以一次生成要跑很多次 U-Net。这就是 DDPM 质量好但采样慢的原因。

## 网络结构

DDPM 用的是 U-Net backbone，后来几乎成了图像 diffusion 的默认骨架:

- 多尺度 encoder-decoder。
- 每个分辨率有卷积 residual blocks。
- 用 GroupNorm。
- 用 sinusoidal timestep embedding 表示时间步。
- 在 $16 \times 16$ feature map 分辨率加入 self-attention。

参数量:

- CIFAR-10 模型约 35.7M。
- LSUN / CelebA-HQ 256x256 模型约 114M。
- 更大的 LSUN Bedroom 模型约 256M。

## 实验结果

### CIFAR-10

DDPM 在无条件 CIFAR-10 上报告:

| Model | IS | FID | NLL |
|---|---:|---:|---:|
| DDPM, true variational bound, fixed variance | 7.67 | 13.51 | <= 3.70 |
| DDPM, simple noise-prediction objective | 9.46 | 3.17 | <= 3.75 |

这里有一个重要张力:

- 优化完整 variational bound，likelihood 更好。
- 优化简化噪声预测目标，视觉样本质量更好。

这也是后来生成模型里很常见的现象: likelihood 好不等于人眼看起来最好。

### 参数化消融

论文比较了直接预测均值和预测噪声。结论很清楚:

> 预测噪声 + simple MSE objective 是样本质量最好的组合。

一些学习方差的配置在这篇论文中还不稳定，后续 Improved DDPM / ADM 才把方差学习处理得更好。

### 高分辨率结果

论文还在 256x256 数据上展示了强结果:

- LSUN Church FID 7.89。
- LSUN Bedroom FID 4.90。
- LSUN Cat FID 19.75。

这些结果说明 diffusion 不只是 CIFAR 上能跑，而是可以扩展到更真实的图像生成。

## 和 DiT 的连接

DDPM 定义了现代 diffusion 的基本训练范式:

```text
加噪 -> 预测噪声 -> 逐步去噪采样
```

DiT 没有推翻这套范式。DiT 主要改了两件事:

| 维度 | DDPM | DiT |
|---|---|---|
| 生成空间 | pixel space | VAE latent space |
| denoiser | U-Net | Transformer |
| 输入形式 | noisy image feature map | noisy latent patch token |
| 核心任务 | predict noise | predict noise |
| 条件注入 | timestep embedding | timestep/class via adaLN-Zero |

所以可以这样串起来:

```text
DDPM: diffusion 训练/采样框架成熟
LDM: 把 diffusion 搬到 latent space，省计算
ViT: 图像可以 patch token 化后交给 Transformer
DiT: 在 latent diffusion 里用 ViT-style Transformer 做 denoiser
```

## 为什么这篇重要

这篇论文让 diffusion 有了一个非常强的工程范式:

- 训练简单。
- 目标稳定。
- 样本质量强。
- 架构可以用 U-Net 承接视觉归纳偏置。
- 后续可以不断改采样器、条件控制、latent space、backbone。

后来 ADM、LDM、Stable Diffusion、DiT、EDM 等，都能在这篇论文的框架上找到根。

## 局限性

1. 采样慢: 1000 步 U-Net 调用成本很高。
2. 初版方差处理还比较粗，部分学习方差实验不稳定。
3. 主要是无条件图像生成，还没有复杂文本条件能力。
4. likelihood 和感知质量不完全一致。
5. 仍然依赖 U-Net 的图像归纳偏置，尚未进入 Transformer scaling 路线。

## 读这篇时抓住什么

1. forward process 是固定加噪，不需要学习。
2. 训练时可以直接构造任意噪声强度的 $x_t$。
3. 网络学的是预测噪声，不是直接画图。
4. 生成是从纯噪声开始，很多小步逐步去噪。
5. simple noise prediction objective 是样本质量的关键。

## 我的评价

综合评分: 10/10。

DDPM 的美感在于它把复杂的生成问题拆成了大量简单的去噪问题。每一步很小、很局部，但连起来就能从噪声里生成图像。这也是后来 diffusion 体系能够爆发的基础。

最值得记住的三句话:

1. 加噪过程把真实图像逐步变成标准高斯噪声。
2. 训练时让模型预测加进去的噪声。
3. 生成时从高斯噪声开始，反复去掉模型预测的噪声。

