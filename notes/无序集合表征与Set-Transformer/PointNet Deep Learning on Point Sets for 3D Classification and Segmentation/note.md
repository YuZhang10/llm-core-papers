## PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation

### 一句话定位

PointNet 是第一类直接以无序 3D point cloud 为输入的深度网络，用 **shared MLP + symmetric max pooling** 解决点集输入的 **permutation invariance** 问题，并统一用于 3D 分类、部件分割和场景语义分割。

### 基本信息

- 论文：**PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**
- 作者：Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
- arXiv：1612.00593
- 时间：2016-12-02
- 任务：
  - 3D object classification
  - 3D object part segmentation
  - indoor scene semantic segmentation
- 输入表示：点集 / 点云，典型为 $N \times 3$，也可带 RGB、法向、归一化坐标等特征。

### 摘要中文翻译

点云是一种重要的几何数据结构。由于点云格式不规则，很多方法会先把它转成规则的 3D voxel grid 或多视角图像，但这会导致数据体积膨胀并带来额外问题。本文设计了一种新的神经网络 PointNet，可以直接消费点云，并尊重点输入的排列不变性。PointNet 为物体分类、部件分割、场景语义解析提供统一架构。虽然结构简单，但效率高、效果强，实验上达到或超过当时主流方法。理论上，论文分析了网络学到了什么，以及为什么它对输入扰动和缺失具有鲁棒性。

### 研究问题

#### 1. 点云为什么难？

点云不是图像网格，也不是 voxel：

| 数据 | 结构 | 常用模型 |
|---|---|---|
| 图像 | 规则 2D grid | CNN |
| 体素 | 规则 3D grid | 3D CNN |
| 点云 | 无序集合 set | 需要 permutation-invariant 模型 |

点云有三个核心性质：

1. **Unordered**
   - 点云是集合，不是序列。
   - 同一个点云有 $N!$ 种输入顺序。
   - 模型输出不应随点顺序改变。

2. **Metric space interaction**
   - 点在欧氏空间中，有距离和邻域关系。
   - 局部结构有语义，例如椅子腿、桌面边缘。

3. **Transformation invariance**
   - 整体平移、旋转、刚体变换不应改变类别或语义。

#### 2. 传统方案的问题

之前方法常把点云转换为：

- 3D voxel grid：计算和存储开销大，分辨率受限。
- multi-view images：依赖视角渲染，丢掉直接 3D 表达。
- handcrafted features：表达能力有限。

PointNet 的基础问题是：

> 如何设计一个直接作用于无序点集、同时可端到端训练的神经网络？

### 核心方法

#### 1. 基础形式：set function approximation

PointNet 用如下形式建模点集函数：

$$
f(\{x_1, \dots, x_n\}) \approx \gamma \left( \max_{i=1}^{n} h(x_i) \right)
$$

其中：

- $x_i$：单个点，通常是 $(x,y,z)$ 或更高维特征。
- $h$：shared MLP，对每个点独立编码。
- $\max$：对所有点做逐维 max pooling，是 symmetric function。
- $\gamma$：后续 MLP，用于分类或生成全局特征。

关键是：

$$
\max(h(x_1), h(x_2), \dots, h(x_n))
$$

对输入点顺序不敏感，因此天然满足 permutation invariance。

#### 2. 分类网络

主线：

1. 输入 $N \times 3$ 点云。
2. T-Net 预测 $3 \times 3$ input transform，对齐输入点。
3. shared MLP 提取 point-wise feature。
4. 可选 feature T-Net 预测 $64 \times 64$ feature transform。
5. shared MLP 映射到高维特征。
6. max pooling 得到 global feature，典型维度 1024。
7. MLP 分类。

简化表示：

$$
X \in \mathbb{R}^{N \times 3}
\rightarrow \text{shared MLP}
\rightarrow \mathbb{R}^{N \times K}
\rightarrow \max_{N}
\rightarrow \mathbb{R}^{K}
\rightarrow \text{MLP}
\rightarrow \text{class logits}
$$

#### 3. 分割网络

分割需要每个点输出标签，因此不仅要 global feature，也要保留 local point feature。

流程：

1. 先提取每个点的局部特征。
2. max pooling 得到全局 shape feature。
3. 将 global feature 复制 $N$ 份，与每个点的局部特征 concat。
4. 再用 shared MLP 输出 per-point label。

即：

$$
\text{per-point feature}_i \oplus \text{global feature}
\rightarrow \text{per-point classifier}
$$

这使每个点的预测同时知道：

- 自己的局部几何；
- 整个物体或场景的全局语义。

例如，同样是细长结构，在椅子上可能是 chair leg，在桌子上可能是 table leg。

#### 4. T-Net：学习对齐

PointNet 引入 mini-network 预测 affine transform：

$$
A \in \mathbb{R}^{3 \times 3}
$$

用于对齐输入点：

$$
x_i' = A x_i
$$

feature transform 类似，但维度更高，例如：

$$
A \in \mathbb{R}^{64 \times 64}
$$

为了避免高维变换破坏信息，加入正交正则：

$$
L_{\text{reg}} = \| I - AA^T \|_F^2
$$

含义：

- 鼓励 $A$ 接近正交矩阵；
- 保持特征空间中的距离和信息；
- 让优化更稳定。

#### 5. 理论解释：为什么 max pooling 有效？

论文证明 PointNet 可以逼近连续 set function。

对定义在点集上的连续函数 $f$，存在连续函数 $h$ 和函数 $\gamma$，使得：

$$
\left| f(S) - \gamma \left( \max_{x_i \in S} h(x_i) \right) \right| < \epsilon
$$

直觉：

- $h$ 学习多个“空间探测器”；
- max pooling 判断某些几何模式是否存在；
- 全局向量记录点云中的关键几何证据。

#### 6. Critical point set

因为 max pooling 每个维度只取最大响应，所以真正决定全局特征的是少数点。

论文定义：

- $C_S$：critical point set，贡献 max pooled feature 的关键点集合。
- $N_S$：upper-bound shape，不改变 max feature 的最大点集范围。

性质：

$$
|C_S| \le K
$$

其中 $K$ 是 max pooling feature 维度。

如果点集 $T$ 满足：

$$
C_S \subseteq T \subseteq N_S
$$

则：

$$
f(T) = f(S)
$$

这解释了 PointNet 对点缺失、噪声、遮挡有一定鲁棒性：只要关键点还在，输出可能不变。

### 关键图表解读

#### 1. Order-invariant 方法比较图

图中比较了三类处理无序点集的方法：

1. **Sequential model / RNN**
   - 把点当序列输入。
   - 问题：顺序本身是人为的，长序列训练困难。

2. **Sorting**
   - 先排序再输入 MLP。
   - 问题：高维空间没有稳定的一维排序。

3. **Symmetry function**
   - 对每个点共享 MLP，再用 symmetric function 聚合。
   - PointNet 采用此路线。

实验结果：

| 方法 | ModelNet40 accuracy |
|---|---:|
| MLP, unsorted input | 24.2 |
| MLP, sorted input | 45.0 |
| LSTM | 78.5 |
| Attention sum | 83.0 |
| Average pooling | 83.8 |
| **Max pooling** | **87.1** |

结论：

> max pooling 是最有效的简单 permutation-invariant 聚合方式。

#### 2. 场景语义分割图

输入是室内 RGB 点云，输出是每个点的语义标签，例如：

- wall
- floor
- table
- chair
- clutter

图中结果显示：

- PointNet 能在真实室内扫描中直接做 point-wise segmentation。
- 即使点云存在缺失、遮挡和采样不均，输出仍较平滑。
- 说明 global feature + per-point feature 的拼接对场景语义有效。

#### 3. Perturbation robustness 图

横轴是 Gaussian noise 标准差，纵轴是分类准确率。

观察：

- 小扰动下准确率下降很慢。
- 噪声 std 增大到约 0.05 后，性能明显下降。
- 到 0.1 左右准确率降到约 30%。

结论：

> PointNet 对小尺度坐标扰动鲁棒，但不是无限制鲁棒；大噪声会破坏关键几何点。

#### 4. ShapeNet part segmentation 图

图中展示 airplane、chair、table、rocket、mug、lamp 等类别的部件分割。

说明：

- PointNet 能对不同类别使用不同部件标签空间。
- 对完整 CAD 和 partial scan 都能预测部件。
- 对细长结构如 chair leg、guitar neck、knife handle 有一定效果，但依赖全局形状理解。

#### 5. CAD retrieval 图

使用 PointNet 提取点云 global feature，可用于检索相似 CAD 模型。

图中：

- 左侧是 query point cloud。
- 右侧是 top-5 retrieved CAD models。

说明 PointNet 学到的 global feature 不只是分类用，也可作为 shape descriptor。

#### 6. Learned point functions / kernels 图

图中类似展示不同 feature dimension 响应的空间区域。

理解：

- 每个维度像一个“几何模式探测器”。
- max pooling 只关心该探测器在点云中是否被强烈激活。
- PointNet 学到的是若干全局几何证据，而非传统 CNN 的局部卷积核。

### 关键贡献

#### 1. 直接处理原始点云

不转 voxel，不转 image，避免：

- voxel 稀疏导致的计算浪费；
- 体素分辨率限制；
- 多视角渲染依赖视角选择。

#### 2. 明确解决 permutation invariance

核心结构：

$$
\gamma \left( \max_i h(x_i) \right)
$$

成为后来 Deep Sets、set learning、point cloud networks 的基础范式之一。

#### 3. 统一分类和分割

同一主干用于：

- object classification：使用 global feature。
- part segmentation：local + global concat。
- scene segmentation：block-wise point labeling。

#### 4. 引入 T-Net 对齐机制

借鉴 Spatial Transformer，但对点云更简单：

- 直接矩阵乘坐标；
- 不需要图像采样和插值；
- 可扩展到 feature alignment。

#### 5. 给出理论分析

提出：

- universal approximation for continuous set functions；
- critical point set；
- upper-bound shape；
- 鲁棒性解释。

### 实验与结论

#### 1. ModelNet40 分类

设置：

- 从 mesh 面上按面积均匀采样 1024 个点。
- 归一化到 unit sphere。
- 数据增强：
  - 随机绕 up-axis 旋转；
  - Gaussian jitter，std = 0.02。

结果：

| 方法 | 输入 | avg. class acc | overall acc |
|---|---|---:|---:|
| 3DShapeNets | volume | 77.3 | 84.7 |
| VoxNet | volume | 83.0 | 85.9 |
| Subvolume | volume | 86.0 | 89.2 |
| MVCNN | image | 90.1 | - |
| Ours baseline | point | 72.6 | 77.4 |
| **PointNet** | point | **86.2** | **89.2** |

结论：

- PointNet 在 3D 输入方法中达到 SOTA 水平。
- 仍略低于多视角图像 MVCNN，可能因为渲染图像捕获了更细表面细节。

#### 2. ShapeNet part segmentation

任务：给每个点预测部件标签。

指标：mean IoU。

结果：

| 方法 | mean IoU |
|---|---:|
| Yi et al. | 81.4 |
| 3D CNN baseline | 79.4 |
| **PointNet** | **83.7** |

部分类别结果：

| 类别 | PointNet IoU |
|---|---:|
| airplane | 83.4 |
| chair | 89.6 |
| guitar | 91.5 |
| laptop | 95.3 |
| mug | 93.0 |
| rocket | 57.9 |

结论：

- PointNet 在多数类别优于传统方法和 3D CNN baseline。
- 对 rocket 等细小复杂结构类别仍较困难。

#### 3. Partial scan 鲁棒性

在 simulated Kinect partial scans 上测试：

| 方法 | complete input | partial input |
|---|---:|---:|
| 3D CNN | 75.3 | 69.7 |
| **PointNet** | **80.6** | **75.3** |

结论：

- 从完整到部分点云，PointNet mean IoU 下降约 5.3。
- 说明它对遮挡和缺失较鲁棒。

#### 4. Stanford 3D semantic parsing 场景分割

输入特征：

$$
[x,y,z,r,g,b,x_{\text{norm}},y_{\text{norm}},z_{\text{norm}}]
$$

即 9D point feature。

训练：

- 按房间切分为 $1m \times 1m$ block。
- 每个 block 训练时随机采样 4096 点。
- 测试时覆盖所有点。

结果：

| 方法 | mean IoU | overall accuracy |
|---|---:|---:|
| handcrafted baseline | 20.12 | 53.19 |
| **PointNet** | **47.71** | **78.62** |

结论：

- 端到端点云学习显著优于手工局部特征 MLP。
- PointNet 可扩展到真实室内大场景。

#### 5. 3D object detection in scenes

基于 semantic segmentation 输出，用 connected component 做 object proposal。

平均 AP：

| 方法 | mean AP |
|---|---:|
| Armeni et al. | 18.22 |
| **PointNet** | **24.24** |

说明 PointNet 的 point-wise semantic output 可支持下游检测。

#### 6. Ablation：输入和特征变换

| Transform | ModelNet40 overall acc |
|---|---:|
| none | 87.1 |
| input transform $3 \times 3$ | 87.9 |
| feature transform $64 \times 64$ | 86.9 |
| feature transform + reg. | 87.4 |
| both | **89.2** |

结论：

- input transform 有稳定收益。
- feature transform 单独不稳，需要正交正则。
- 两者结合最好。

#### 7. 复杂度

| 方法 | 参数量 | FLOPs/sample |
|---|---:|---:|
| PointNet vanilla | 0.8M | 148M |
| **PointNet** | **3.5M** | **440M** |
| Subvolume | 16.6M | 3633M |
| MVCNN | 60.0M | 62057M |

结论：

- PointNet 远比 3D CNN 和 multi-view CNN 高效。
- 时间和空间复杂度对点数 $N$ 近似线性：

$$
O(N)
$$

### 局限性

1. **缺少显式局部结构建模**
   - 每个点先独立过 shared MLP。
   - 点与邻域的关系主要靠后续全局 max 间接捕获。
   - 对细粒度局部几何不如后来的 PointNet++。

2. **max pooling 是信息瓶颈**
   - 每个 feature dimension 只保留最大响应点。
   - 对密集几何细节、纹理级结构保留不足。

3. **旋转不变性不是严格保证**
   - T-Net 学习对齐，但不等于数学上的 rotation invariance。
   - 对任意 3D rotation 仍可能敏感。

4. **对点密度和采样策略有依赖**
   - 训练常用均匀采样。
   - 真实传感器点云存在非均匀密度、遮挡、噪声。

5. **全局分类强，局部关系弱**
   - 对物体级分类很简洁有效。
   - 对复杂场景中的对象边界、局部交互，需要更强邻域建模。

### 放进大模型基础知识体系里怎么理解

#### 1. PointNet 是 set modeling 的经典架构

它回答的是基础问题：

> 当输入 token 没有顺序时，神经网络该如何设计归纳偏置？

文本 Transformer 依赖 positional encoding，因为语言有顺序。

点云是 set，所以需要：

- permutation invariance：分类任务；
- permutation equivariance：分割任务。

分类中：

$$
f(\pi X) = f(X)
$$

分割中，如果输入点被重排，输出也应对应重排：

$$
F(\pi X) = \pi F(X)
$$

PointNet 的 shared MLP + pooling 正好满足这种结构。

#### 2. 它是 “token-wise encoder + global pooling” 范式

可类比到大模型中的：

- token embedding；
- shared feed-forward / encoder；
- pooling / CLS token；
- global representation；
- downstream head。

区别是：

- PointNet 没有 self-attention；
- 点之间没有显式 pairwise interaction；
- 主要靠 max pooling 做全局聚合。

#### 3. 它体现了 inductive bias 的重要性

如果直接把点云拍平成向量输入 MLP，效果很差，因为破坏了集合对称性。

PointNet 的成功不是因为模型很大，而是因为结构满足数据本质：

- 点云无序；
- 点共享同一几何空间；
- 任务需要变换鲁棒性。

这对理解大模型也重要：

> 架构不是中性的；合适的对称性和不变性会显著降低学习难度。

#### 4. 它是后续 3D foundation model 的底层积木

后续很多方法都继承或改进 PointNet：

- PointNet++：加入局部邻域层级结构。
- DGCNN：用动态图卷积建模点间关系。
- Point Transformer：引入 attention。
- 3D pretraining / point cloud foundation models：仍需要点级 embedding、局部-全局聚合、set invariance。

### 我需要记住什么

1. PointNet 的核心公式：

$$
f(S) = \gamma \left( \max_{x_i \in S} h(x_i) \right)
$$

2. 用 **symmetric function** 解决点云输入的 **permutation invariance**。

3. 分类靠 global feature；分割靠：

$$
\text{local point feature} \oplus \text{global shape feature}
$$

4. T-Net 学习输入和特征对齐，feature transform 用正交正则：

$$
L_{\text{reg}} = \| I - AA^T \|_F^2
$$

5. max pooling 不只是工程技巧，也带来理论解释：
   - critical points 决定输出；
   - 非关键点缺失不一定改变结果；
   - 因此对缺点、遮挡、小扰动鲁棒。

6. PointNet 的最大短板：
   - 没有显式局部邻域建模。
   - 这直接催生 PointNet++。

7. 一句话背诵版：

> PointNet 用 shared MLP 提取每个点特征，再用 max pooling 聚合成全局特征，从而直接在无序点云上实现分类和分割，是点集深度学习的基础架构。
