# Tool-use Read Next

这个文件用来记录后续想顺着看的 tool-use / agent tool orchestration 演变线。当前直觉是：今天看起来标准的“生成 plan、读工具描述、选择工具、执行、把结果写回上下文”，在 2022-2024 之间逐步从论文原型变成工程接口。

## 我想追的问题

- LLM 什么时候开始被明确当成 tool controller / router？
- 工具描述从自然语言 prompt，怎么演变成 schema、function calling、MCP 这类接口？
- tool-use 和 ReAct、HuggingGPT、Voyager 的关系分别是什么？
- planner、tool selector、executor、verifier、memory/state 分别在哪些论文里被拆出来？
- 当前标准范式到底继承了哪些早期论文，哪些只是 product/API engineering 的结果？

## 演变主线

### 1. 早期系统观：LLM 不必自己做所有事

核心思想：

```text
LLM 负责理解、推理、路由；
外部模块负责搜索、计算、数据库、专业模型或环境动作。
```

建议读：

- **WebGPT**：用浏览器检索来回答问题，是“LLM + 外部信息源”的早期重要系统。
- **MRKL Systems** (`2205.00445`)：把 LLM 和 symbolic tools / expert modules 组合成模块化系统。
- **ReAct** (`2210.03629`)：把 `Thought -> Action -> Observation` 变成清晰的交互循环。

这里重点看：工具调用不只是 API 调用，而是 agent state 会被 observation 更新。

### 2. 工具使用变成模型能力：什么时候、调用什么、怎么用结果

核心思想：

```text
模型不仅要会调用工具，还要学会：
什么时候调用、传什么参数、如何把工具结果纳入后续推理。
```

建议读：

- **Toolformer** (`2302.04761`)：让模型通过自监督数据学会调用 calculator、search、QA、calendar 等 API。
- **PAL / Program-aided LM** (`2211.10435`)：把计算步骤交给 Python 程序执行。
- **Program of Thoughts** (`2211.12588`)：把推理和计算分离，让程序承担精确计算。

这里重点看：tool-use 不一定是 agent 系统，也可以是模型生成过程中插入 API/program call。

### 3. 2023 系统爆发：LLM 作为多工具/多模型总控

核心思想：

```text
用户请求 -> 任务拆解 -> 工具/模型选择 -> 执行 -> 结果汇总
```

建议读：

- **HuggingGPT** (`2303.17580`)：ChatGPT 作为 controller，调度 Hugging Face 专家模型。
- **Visual ChatGPT** (`2303.04671`)：ChatGPT 调用视觉基础模型，完成图像理解、绘制和编辑。
- **Chameleon** (`2304.09842`)：把多种工具组合成 plug-and-play 的推理系统。
- **ViperGPT** (`2303.08128`)：用 Python 程序调用视觉模块完成视觉推理。

这里重点看：HuggingGPT 的创新不是单次 tool call，而是 task graph、resource dependency、model description 和 execution log。

### 4. API 调用能力评测与大规模工具库

核心思想：

```text
工具变多之后，难点从“能不能调用一个工具”变成：
能不能选对 API、填对参数、处理调用失败、完成多步任务。
```

建议读：

- **API-Bank** (`2304.08244`)：面向 tool-augmented LLM 的评测基准。
- **Gorilla** (`2305.15334`)：连接大量 API，关注 API 调用准确性和 hallucination。
- **ToolLLM** (`2307.16789`)：面向 16000+ 真实 API 的工具学习和调用。

这里重点看：tool description 从“给模型看的说明文字”，逐渐变成需要评测、检索、约束和对齐的能力接口。

### 5. 工程标准化：function calling / structured outputs / agent platform

核心思想：

```text
工具调用从 prompt convention 变成 API contract。
```

产品/API 里程碑：

- **ChatGPT Plugins**，OpenAI，2023-03-23：把浏览、代码解释器和第三方服务作为 ChatGPT 工具接入。
- **OpenAI Function Calling**，2023-06-13：工具名、描述、JSON schema、模型选择调用，成为工程上很关键的标准形态。
- **Structured Outputs**，2024：更严格地保证函数参数匹配 JSON Schema。
- **MCP**，Anthropic，2024-11-25：把工具/数据源连接抽象成更通用的协议。
- **Agents SDK / tracing / tool platform**，2025：把工具、状态、追踪、工作流进一步平台化。

这里重点看：今天觉得“标准”的 tool-use，很大一部分来自 API 和 SDK 把早期论文里的 prompt protocol 固化成工程接口。

## 和当前 Agent Memory Layer 里的论文怎么接

- **ReAct**：定义在线 action-observation 交互。
- **HuggingGPT**：定义结构化工具/模型编排的 working memory：task list、dependency、resource、execution result。
- **Reflexion**：工具执行失败后，把失败归因写成 verbal memory。
- **Voyager**：把成功工具调用/代码沉淀成 skill library，也就是 procedural memory。
- **AutoGen / MetaGPT**：把 tool-use 扩展到多 agent 协作、角色分工和消息协议。

一句话：

```text
HuggingGPT 是 tool orchestration；
Voyager 是 tool/skill accumulation；
AutoGen/MetaGPT 是 tool-use + multi-agent workflow。
```

## 之后读的时候重点记录

- 工具描述是什么形式：自然语言、JSON schema、OpenAPI、MCP server。
- 谁负责规划：单个 LLM、planner agent、固定 workflow、搜索算法。
- 谁负责选择工具：LLM 直接选、retriever 召回、router 分类、学习到的 policy。
- 工具结果怎么回到上下文：observation、execution log、structured state、memory buffer。
- 有没有 verifier：执行错误、环境反馈、单元测试、LLM critic、人类确认。
- 是否形成长期能力：只解决当前任务，还是沉淀 reflection、skill、workflow template。

## 优先阅读顺序

1. ReAct
2. MRKL Systems
3. Toolformer
4. HuggingGPT
5. API-Bank / Gorilla / ToolLLM
6. OpenAI function calling / structured outputs 文档
7. MCP
8. AutoGen / MetaGPT / Voyager 回看它们如何使用工具状态和技能记忆

