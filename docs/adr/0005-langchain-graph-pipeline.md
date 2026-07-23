# ADR 0005: 管线编排迁移至 LangGraph + LCEL

**日期**: 2026-07-01
**状态**: 已实施
**影响范围**: `pipeline/orchestrator.py`, `modules/intervention/scale/scorer.py`, `modules/intervention/generator.py`

## 上下文

当前四阶段管线（Safety → Emotion → Router → Intervention）由 `pipeline/orchestrator.py` 中手写的 `run_pipeline()` 函数编排，通过 `if/else` 控制条件短路。这种实现功能正确但：

- 管线拓扑隐式编码在条件分支中，新人需要阅读完整函数才能理解流程
- 缺乏标准化的图结构可视化能力
- 项目目前使用 LangChain 仅作为消息格式 + HTTP 客户端，未发挥其在编排层的价值

同时量表计分器使用正则从 LLM 自由文本回复中提取数字，对输出格式的容错性依赖正则健壮性；干预生成器中三个 `generate_*` 方法存在重复的消息构建模式。

## 决策

### 1. 管线编排 → LangGraph StateGraph

用 `langgraph.StateGraph` 替换手写编排逻辑：

```
          ┌── blocked? / level>=3 → crisis_emotion → crisis_route ──┐
          │                                                          ↓
[safety] ─┤                                                    [intervention]
          │                                                          ↑
          └── pass → [emotion] → [router] ──────────────────────────┘
```

- 定义 `PipelineState` TypedDict 作为图状态
- 4 个 node 函数 + 1 条条件边实现安全短路
- `run_pipeline()` 和 `run_video_pipeline()` 保留为图入口，签名不变
- 视频管线复用同一张图，通过初始 state 注入预处理数据

### 2. 量表计分 → PydanticOutputParser

用 `langchain_core.output_parsers.PydanticOutputParser` 替换正则提取：

- 定义 `ScaleScoreResult(score: int)` 模型
- LLM 直接输出 `{"score": 2}` 结构，Parser 自动解析
- 消除正则鲁棒性问题（"2分"、"给2"、"2." 等变体）

### 3. 干预生成 → LCEL

用 LangChain Expression Language 消除三个 `generate_*` 方法中的消息构建重复：

```python
chain = ChatPromptTemplate.from_messages([...]) | llm | StrOutputParser()
```

### 4. 不改动的边界

- 对外接口（`run_pipeline()`、`score()`、`generate_*()`）签名和返回值**完全不变**
- 独立模块调试端点（`/api/v1/modules/*`）不走管线，不受影响
- 合约模型（`schemas/contracts/v1.py`）不变
- 全部现有测试入口不变

## 后果

### 正面

- **可读性**：管线拓扑显式表示为有向图，可视化与调试更直观
- **可扩展性**：后续新增节点（如 A/B 路由、人机协作）仅需添加图节点和边
- **计分可靠性**：结构化输出消除正则提取的边界情况
- **代码简洁**：消除手写编排 ~100 行，消除生成器重复 ~20 行

### 风险与缓解

- **新依赖 langgraph**：作为 LangChain 生态的官方包，成熟度可接受
- **图调用开销**：多一层 `invoke` 包装，延迟增幅可忽略（非实时系统）
- **回滚**：`backup/pre-langchain` 分支保留当前状态，可随时切回
