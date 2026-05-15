# MTGR 讨论笔记：它到底是不是 Generative Recommendation？

日期：2026-05-12

这篇笔记记录一次围绕 MTGR 的讨论。主线不是复述论文，而是澄清一个核心问题：

> MTGR 最重要的地方，到底是“生成式推荐”，还是把传统精排改造成 request-level multi-candidate tokenized ranker？

我的结论是：**MTGR 狭义上并不是 OneRec 那种直接生成 item 的生成式推荐模型。它更像是一个 HSTU 化、token 化、request-level 的工业精排模型。**

换句话说，它不是：

```text
输入用户历史 -> 从 item vocabulary 里生成下一个 item
```

而是：

```text
输入用户特征、历史行为、实时行为、已给定候选及 cross features
-> 一次输出每个候选的点击/转化 logit
```

这个判断会贯穿下面的分析。

## 1. 先把 DLRM 和 cross features 讲清楚

DLRM 可以简单理解成工业推荐排序模型的经典范式：给一个用户和一个候选 item，预测这个用户对这个 item 的点击、转化或购买概率。

传统 pointwise 精排通常是：

```text
D_i = [U, S, R, I_i, C_i]
```

其中：

- `U` 是用户特征，例如年龄、城市、会员等级、长期偏好。
- `S` 是历史行为序列，例如过去 30 天点击、购买、浏览过的商家或商品。
- `R` 是实时行为序列，例如当前 session 或最近几分钟、几小时的搜索、点击、加购。
- `I_i` 是第 `i` 个候选 item 自身特征，例如 item_id、shop_id、类目、品牌、价格、距离、配送时间。
- `C_i` 是第 `i` 个候选对应的 cross features，即 `cross(user, item_i, context)`。

cross features 的一句话定义：

> cross features 是由多个原始特征组合出来、专门刻画 user/context 与 candidate 之间关系的特征。

它不是单独的用户特征，也不是单独的 item 特征，而是交互特征。

例子：

```text
user x item:
用户历史上是否点击/购买过这个 item

user x shop:
用户过去 30 天对这个商家的点击率、下单率、曝光次数

user x category:
用户对“奶茶/火锅/快餐”类目的历史偏好

item x context:
这个商品在当前时间段、当前位置、当前天气下的 CTR

user x item x context:
这个用户在晚上、当前位置、对这个商家类型的转化统计
```

在工业推荐里，cross features 往往非常强。它们来自长期特征工程、统计计数、实时特征系统和业务先验。MTGR 的核心立场就是：**生成式推荐想 scale 可以，但不能把这些工业排序里最有用的 cross features 扔掉。**

## 2. 传统 pointwise DLRM 怎么做

最经典的精排方式是对每个候选单独构造一条样本：

```text
[user, history, realtime, item_1, cross_1] -> logit_1
[user, history, realtime, item_2, cross_2] -> logit_2
[user, history, realtime, item_3, cross_3] -> logit_3
...
```

这个范式有一个明显问题：同一个 request 里，`user`、`history`、`realtime` 对所有候选几乎是一样的，却被重复编码了很多次。

如果精排阶段有 `K` 个候选，传统 pointwise DLRM 近似要做 `K` 次用户侧和交互侧计算。

这导致 scaling dilemma：

- 扩大 user module：用户表示更强，且用户侧计算可以复用，但用户和具体 candidate 的细粒度交互不足。
- 扩大 cross module 或 MLP：用户和 candidate 的交互更强，但每个候选都要算，成本随候选数线性增长。

所以传统精排想要大模型化，会被推理延迟和训练成本卡住。

## 3. MTGR 的关键重排

MTGR 把同一个用户或同一个 request 下的多个候选聚合成一个样本：

```text
[
  user,
  history,
  realtime,
  [item_1, cross_1],
  [item_2, cross_2],
  [item_3, cross_3],
  ...
] -> [logit_1, logit_2, logit_3, ...]
```

这个是我们讨论里反复确认的重点。

它不是把 candidate 作为 next token 生成出来，而是把已经给定的 candidates 作为输入的一部分，然后一次输出所有候选的分数。

因此它和 LLM next-token prediction 不一样。

LLM 是：

```text
输入前文 tokens
-> 输出整个词表上的下一个 token 概率分布
```

MTGR 是：

```text
输入用户 token + 历史 token + 实时 token + 已给定候选 token
-> 输出每个候选 token 的点击/转化 logit
```

候选不是被生成的输出，而是被打分的输入。

## 4. “全部 token 化”到底是什么意思

这里最容易被 LLM 语境误导。MTGR 里的 token 不是 NLP tokenizer 里的词，也不是 OneRec 里要生成的 item id。

更准确地说：

> token = 一个统一维度的 dense feature block。

MTGR 会把原来 DLRM 里的各种特征块，整理成一串 `d_model` 维向量，然后让 HSTU 在这些向量之间做 attention。

假设一个外卖 request 有用户和 3 个候选商家：

```text
用户特征 U:
city=北京, age_bucket=25-30, member_level=gold

历史行为 S:
过去买过 火锅店X、奶茶店Y、炸鸡店Z

实时行为 R:
刚刚搜索了“奶茶”
刚刚点开了“喜茶”

候选 A 自身特征 I_A:
shop_id=A, category=咖啡, distance=800m, delivery_time=25min, price=35

候选 A 的 cross features C_A:
用户过去 30 天点过咖啡类目 8 次
用户过去 30 天买过这个商家 2 次
用户对该商家曝光 10 次点击 1 次
当前时间段 x 咖啡类目 x 用户城市的 CTR
```

传统 DLRM 是每个候选单独做：

```text
[U, S, R, I_A, C_A] -> logit_A
[U, S, R, I_B, C_B] -> logit_B
[U, S, R, I_C, C_C] -> logit_C
```

MTGR 则把它变成一串 token：

```text
[
  user_city_token,
  user_age_token,
  user_member_token,

  history_item_1_token,
  history_item_2_token,
  history_item_3_token,

  realtime_item_1_token,
  realtime_item_2_token,

  candidate_A_token,
  candidate_B_token,
  candidate_C_token
]
```

其中最关键的是 candidate token：

```text
candidate_i_token =
MLP(concat(
  embedding(item_i_features),
  embedding(cross(user, item_i))
))
```

也就是说，candidate token 不是只有 `item_id`，而是：

```text
candidate_i_token = [item_i 自身特征 + user-item_i cross features] 的统一向量表示
```

所以 cross features 没有丢，而是被折进 candidate token 里。

## 5. “压缩”到底压缩了什么

MTGR 里的压缩不是说“不再给每个 item_id 一个 token”，也不是压缩 item vocabulary。

它压缩的是样本和计算：

传统 pointwise 精排：

```text
user + item_1 -> forward -> logit_1
user + item_2 -> forward -> logit_2
user + item_3 -> forward -> logit_3
...
```

MTGR：

```text
[user tokens, history tokens, realtime tokens, candidate_1 token, ..., candidate_K token]
-> one forward
-> [logit_1, ..., logit_K]
```

也就是说，同一用户的一批候选被聚合成一个 request-level 样本，用户侧信息只编码一次。

它节省的是：

```text
重复编码用户侧 + 长序列 + 大交互模块
```

它没有节省的是：

```text
每个 candidate 的 cross feature 构造
```

`cross_i` 仍然要为每个候选提前取数、embedding、拼接，再做成 candidate token。

## 6. 为什么纯 next-token generative rec 不自然使用 cross features

纯 next-token generative rec 的形式通常是：

```text
输入：用户历史行为
输出：下一个 item token / item id
```

candidate 是“要被生成的输出”，不是预先给定的输入。

但 cross feature 恰恰需要 candidate 已经确定：

```text
cross(user, candidate_i)
```

比如：

```text
用户过去 7 天对这个商家的点击率
用户和这个 item 所属类目的交互次数
用户当前位置到这个商家的距离分桶
```

这些特征必须知道候选是谁才能计算。

如果生成模型每一步要在全量 item vocabulary 上预测下一个 item，同时又要使用 cross features，那理论上需要对每一个可能 item 都计算：

```text
cross(user, item_1)
cross(user, item_2)
...
cross(user, item_N)
```

这在工业规模下非常贵，也破坏了生成式模型直接从 vocabulary 生成 item 的形式。

所以纯生成式推荐通常更依赖 item id、semantic id、行为序列 token，而不容易直接接入传统 DLRM 里 candidate-specific 的强交叉特征。

MTGR 的处理方式是：候选已经给定，把 `[item_i, cross_i]` 做成 candidate token，再打分。

这也是为什么它更像排序模型，而不是严格意义上的 item generative model。

## 7. 历史行为和实时行为的区别

历史行为 `S`：相对稳定、较早发生的长期行为。

例如：

```text
过去 30 天点击过的商家
过去 90 天购买过的品类
长期喜欢的口味、价格带、品牌
```

实时行为 `R`：离当前请求很近、反映即时兴趣的行为。

例如：

```text
用户刚刚搜索了“奶茶”
用户刚点开了一个炸鸡店
用户最近 5 分钟连续浏览了几家火锅
用户当前 session 里的点击、加购、返回
```

实时行为非常有价值，但更容易产生信息泄漏。

例如训练聚合窗口里，下午 3 点的候选不能看到晚上 8 点用户点击了什么。否则模型就是偷看未来。

所以 MTGR 设计了 dynamic mask：

- 静态用户信息和历史序列对所有 token 可见。
- 实时行为按时间因果可见。
- candidate token 只看自己，不看其他 candidate。

这不是标准 causal mask，而是推荐场景下的时间感知 mask。

## 8. 能不能不用 HSTU，只用经典精排模型或 MLP

这是讨论里非常关键的问题。

用户提出的问题是：

> 如果我搞一个经典的精排模型，样本就这样设计，是不是也可以？
>
> ```text
> [
>   user,
>   history,
>   realtime,
>   [item_1, cross_1],
>   [item_2, cross_2],
>   [item_3, cross_3],
>   ...
> ] -> [logit_1, logit_2, logit_3, ...]
> ```
>
> 这里根本没有需要用到序列模型和生成的概念，也不需要 token，对吧？

我的判断是：**理论上完全可以，而且这个思想本身并不新。**

如果候选数固定、历史长度也截断固定，直接 flatten 后接一个 MLP，是可以训练的：

```text
flatten(user, history, realtime, item_1+cross_1, ..., item_K+cross_K)
-> MLP
-> K logits
```

这可以看成一个固定 K 的 listwise MLP ranker。

但它会遇到几个问题。

### 8.1 MLP 不天然知道候选是重复结构

对人来说：

```text
[item_1, cross_1], [item_2, cross_2], [item_3, cross_3]
```

是同一种候选结构重复 `K` 次。

但普通 MLP flatten 后看到的是：

```text
x_100 到 x_200 是 candidate_1
x_201 到 x_301 是 candidate_2
x_302 到 x_402 是 candidate_3
```

它不会自动共享“候选打分逻辑”。candidate_1 和 candidate_2 的参数天然绑定在不同输入位置上。

合理的排序模型应该具备一种性质：如果候选顺序打乱，输出 logit 顺序也相应打乱。但普通 flatten MLP 不保证这个排列等变性。

### 8.2 MLP 很难高效建模长历史和每个候选的交互

精排最关键的是：

```text
candidate_i 和 history 中哪些 item 相关？
candidate_i 和 realtime 行为是否匹配？
candidate_i 的 cross_i 如何结合用户长期兴趣？
```

如果历史有 1000 个 item，候选有 100 个，真正需要的是类似：

```text
candidate_i attend to history_j
```

这种交互具有大量重复结构和共享模式。Attention/HSTU 很适合做这个。

MLP 理论上可以学，但要靠参数硬拟合，效率和泛化都不理想。

### 8.3 MLP 对可变长度和 mask 不自然

实际系统里会有：

```text
history 长度不固定
realtime 长度不固定
候选数可能不同请求不同
候选来自不同召回通道
训练窗口里候选时间不同
```

候选数在精排阶段可以固定 top-K，这点不是核心难点。真正麻烦的是历史和实时行为的可变长度，以及实时行为的时间泄漏。

MLP 当然可以 padding，但 mask 怎么参与 MLP 并不自然。最后往往还是要引入 pooling、attention、set module 或 shared candidate scorer。

一旦引入这些模块，就已经接近 MTGR 的路线。

## 9. 所以 multi-candidate request-level scoring 是不是新想法

不是。

把所有 candidates 放到一个 request 里一次打分，以节省 user 部分计算量，这个思想本身不需要 HSTU，也不需要生成式模型。

推荐系统里类似思想早就存在：

- user tower 复用
- batch scoring
- listwise rerank
- slate ranking
- 多目标共享 user representation

它们本质上都在做：

```text
用户侧算一次，多个候选共享
```

所以 MTGR 的 novelty 不应该被理解成：

> 第一次想到把多个 candidate 一起算。

更准确的理解是：

> 在保留工业 DLRM 全量特征，尤其是 candidate-specific cross features 的前提下，把多个候选组织成 token 序列，用 HSTU 一次性建模，并证明这个框架可以随着层数、维度、序列长度继续 scale，且能在线主流量跑起来。

## 10. MTGR 相比普通 user tower 的区别

普通 user tower 常见做法是：

```text
user_repr = BigUserModel(user, history, realtime)

for each candidate_i:
    logit_i = SmallModel(user_repr, item_i, cross_i)
```

它的好处是用户侧大模型只算一次。

但问题是：大模型阶段没有看到具体 candidate 和 cross feature。很多细粒度交互只能交给后面的小模型。

MTGR 则是：

```text
candidate_i_token = MLP([item_i, cross_i])

HSTU([
  user_tokens,
  history_tokens,
  realtime_tokens,
  candidate_1_token,
  ...,
  candidate_K_token
])
-> candidate_i_hidden
-> shared head
-> logit_i
```

也就是说，candidate token 在大模型内部就可以和用户、历史、实时行为交互。

它想 scale 的不是纯 user tower，而是带 cross feature 的排序交互部分。

## 11. MTGR 的真实亮点

经过讨论，我认为这篇文章的真实亮点不是“conceptual novelty”，而是“工程组合很硬”。

它把下面几件事放到一起：

```text
request-level multi-candidate scoring
+ DLRM 全量特征保留
+ candidate-specific cross features 折进 candidate token
+ 长历史和实时行为 token 化
+ HSTU 做统一交互
+ GLN 处理异构 token 分布
+ dynamic mask 防止实时行为泄漏
+ TorchRec 训练系统优化
+ 工业主流量 A/B 验证
```

它最聪明的地方是抓住了一个工业矛盾：

```text
纯生成式推荐容易 scale，但丢 cross features。
传统 DLRM 有 cross features，但 scale cross module 太贵。
```

MTGR 的答案是：

```text
把 user/history/realtime/candidate+cross 全部 token 化，
用一个可扩展 encoder 做统一交互，
一次输出多个 candidate logits。
```

所以我更愿意把它叫作：

```text
Request-level tokenized ranker
```

或者：

```text
GRM-scalable DLRM
```

这个命名比 “Generative Recommendation” 更贴近它实际在做的事情。

## 12. 对 “Generative Recommendation” 这个命名的看法

如果按狭义定义，生成式推荐应该是：

```text
用户历史 -> 生成 item id / semantic id / item token
```

那 MTGR 不算严格的 generative recommendation，因为它没有从 item vocabulary 里生成 candidate。

如果按广义定义，把 GRM 看成：

```text
tokenized recommendation data
+ Transformer/HSTU backbone
+ user-level compression
+ long behavior sequence modeling
```

那 MTGR 可以算一种 generative recommendation inspired architecture。

但我认为更准确的评价是：

> MTGR 借用了 GRM/HSTU 的 token 化和 scaling 思路，改造了传统工业精排，而不是做了一个真正的 item generation model。

## 13. 后续值得继续追问的问题

### 13.1 candidate 之间完全不互相看是否合理

MTGR 为了避免泄漏，让 candidate token 只看自己，不看其他 candidate。

这对 pointwise ranking 是安全的，但也放弃了 slate/listwise 信息。

可以追问：

- 如果做最终重排，candidate 之间是否应该互相看？
- 是否可以区分训练标签泄漏和合法的候选集合上下文？
- 是否存在一种 mask，让 candidate 之间共享无标签的竞争信息，但不泄漏行为结果？

### 13.2 cross features 的消融是否公平

论文显示去掉 cross features 后性能大幅下降，这支持了作者观点。

但还可以追问：

- 去掉 cross features 后，是否给模型足够时间和规模重新收敛？
- cross features 中是否包含非常强的统计特征，导致其他模型难以公平替代？
- 如果用更强的 representation learning 替代 cross features，差距会缩小吗？

### 13.3 HSTU 的必要性到底有多强

可以设计几个 baseline：

```text
request-level flatten MLP
request-level shared candidate scorer
request-level DCN / DeepFM
request-level Transformer
request-level HSTU
```

比较这些模型后，才能更清楚地回答：

> MTGR 的收益到底来自 request-level 聚合，还是来自 HSTU？

### 13.4 它和 OneRec 的差别

OneRec 更像：

```text
用户历史 -> 直接生成推荐序列
```

MTGR 更像：

```text
用户 + 已召回候选 + cross features -> 多候选统一打分
```

所以 OneRec 的野心是统一 retrieve and rank，MTGR 的野心是把成熟 DLRM 工业特征体系升级成可 scaling 的大模型排序框架。

这两篇读在一起，很适合形成一个判断：

> 工业推荐里的“生成式”不是一条单线路线，而是一组围绕 tokenization、long sequence modeling、user-level compression、candidate scoring 的范式迁移。

