# 路由模块：分层规则引擎 + 情绪修正矩阵，LLM 仅做降级补充

智能路由模块需要将每个请求分派到四种干预路径之一（general / knowledge / comfort / crisis），按风险从低到高排列。我们决定采用分层规则引擎作为主决策机制，以 risk 为第一优先级、安全等级和情绪类型作为修正因子；LLM 仅作为低置信度时的可选降级补充，不参与主决策。

## 为什么

### 为什么不用纯 LLM 路由

- 三个路由目标是穷举且互斥的，规则引擎可以覆盖全部决策空间，不需要 LLM "发现"新模式。
- 心理干预场景对可解释性和确定性有要求。如果系统路由到 crisis，必须能说清原因——规则引擎每一步都有审计痕迹，LLM 的 reasoning 不可靠。
- 延迟和成本。每条请求都要过一次 LLM 对成本敏感，且网络延迟不可控。
- 项目已经有 TherapyChain 承担 LLM 交互职责，Router 不需要重复。

### 为什么分层而不是单一维度

- risk 已经是 Emotion 模块综合了情绪类别、安全等级、多模态冲突后的聚合信号，适合做第一优先级。
- 但 risk 无法区分"我今天心情还行，什么是利弊分析？"（信息提问）和普通的日常闲聊——两者 risk 都接近零，前者应走 knowledge 科普路线，后者走 general 闲聊。因此引入 intent 覆盖层。
- 相同 risk 值下，不同情绪类型需要不同干预策略：anger + 高强度不宜直接安抚，sadness 需要共情而非说教。
- 安全过滤的 level >= 2 是危险信号，应在路由器中触发升段。
- 因此 risk 定段位、intent 覆盖纯信息提问、safety 可升段、emotion 做段内微调的分层模型最契合项目需求。

### 为什么无状态

- 会话上下文感知是 Intervention 模块的职责（通过 TherapySessionMemory 和 SessionManager）。
- Router 保持"当前轮独立裁决"使每轮路由可独立审计和复现。
- 如果需要趋势信号，更好的做法是在 Emotion 模块的 risk 计算中加入历史因子，而非在 Router 层引入状态。

## 分层决策模型

```
第一优先：risk（0~1）
  ├─ risk >= 0.7  → 高危段（crisis）
  ├─ risk >= 0.5  → 中危段（comfort）     ← 中高风险先安抚
  ├─ risk >= 0.1  → 低位段（knowledge）   ← 低风险给知识科普
  └─ risk <  0.1  → 底段（general）       ← 无风险闲聊

第二修正：intent（意图覆盖）
  └─ general + intent=information → knowledge
     纯信息提问（如"什么是利弊分析"）不走闲聊，走知识科普路线
     仅在 risk 为 general 段且无情绪信号时生效

第三修正：safety.level
  └─ level >= 2   → 升一段（general→knowledge，knowledge→comfort，comfort→crisis）
                     注：level >= 2 触发安全短路，不经过 Router 的 risk 判定阶段

第四修正：primary_emotion + intensity
  └─ 同段位内，情绪偏向影响 reason 标签和 confidence，不跨段改变 route
```

**段位否决原则**：risk 决定的段位具有最终否决权。情绪偏向不能将 knowledge 段升为 comfort，也不能将 comfort 段降为 knowledge。意图覆盖仅在 risk 最低段（general）时生效，不跨段。情绪偏向仅在段位方向一致时取消防信度惩罚。

## 情绪修正矩阵

| primary_emotion | 修正倾向 | 说明 |
|-----------------|---------|------|
| neutral | 保持基础路由 | 无额外信号，置信度不扣分 |
| anxiety | 偏向 comfort | 焦虑需要安抚和确定感 |
| sadness | 偏向 comfort | 需要共情而非说教 |
| anger | 偏向 knowledge | 愤怒时直接安抚可能激化 |
| anger (intensity > 0.7) | 反转为 comfort | 高愤怒时讲道理反而激化 |
| fear | 偏向 comfort | 恐惧先需要安全感 |
| stress | 偏向 knowledge | 需要应对策略 |
| happiness | 保持基础路由 | 正常状态，不干预 |
| confusion | 偏向 knowledge | 需要信息澄清 |

## LLM 降级规则

```
触发条件（满足任一）：
  1. confidence < 0.7   → 规则引擎判断自己不够确定
  2. mixed_signals = True → 多模态信号打架，需要语义理解

触发后行为：
  → LLM 可用：规则引擎结果与 LLM 建议对比
    - 一致：升 confidence 到 0.85
    - 不一致：以规则引擎为准，meta 标注 llm_disagree
  → LLM 不可用/超时/报错：静默跳过，规则引擎独立决策

未配置 LLM 时：降级机制不启用，规则引擎始终独立决策
```

Router 接收一个可选的 LLM 适配器参数，由工厂注入。没有配置 LLM 时降级自动跳过。

## 路由置信度

confidence 从 1.0 起步，以下因素依次扣分后截断到 [0.5, 1.0]：

| 扣分项 | 扣分 | 条件 |
|--------|------|------|
| 边界极近 | -0.15 | risk 距阈值 < 0.05 |
| 边界较近 | -0.08 | risk 距阈值 < 0.10 |
| 安全升段 | -0.05 | safety.level=1 触发了升段修正 |
| 情绪修正 | -0.05 | 情绪偏向与 risk 段位方向冲突 |
| 混合信号 | -0.10 | modality_notes 中 mixed_signals=True |

情绪偏向与 risk 段位方向一致时（如 confusion 偏向 knowledge 且当前在 knowledge 段），不扣情绪修正分。

## 容错

- 规则引擎本身无外部依赖，不会出错。
- LLM 降级调用超时或失败时，静默回退到规则引擎独立决策，并在 meta 中记录放弃原因。
- Router 不成为单点故障。

## 考虑过的方案

### 纯 LLM 路由
每条请求调用 LLM 做 few-shot 分类。优点是语义理解强、边界柔和。缺点是延迟和成本不可控、推理不可审计、对于三选一问题过度复杂。不选。

### 风险单一维度（当前 Stub 方案）
只依赖 emotion.risk 做阈值路由，忽略 primary_emotion 和安全等级。优点是简单。缺点是丢掉了大量可用信号——同等 risk 的 anger 和 sadness 显然不应走同一路径。不选。

### 有状态 Router
Router 内建会话历史感知，综合多轮情绪趋势做路由。优点是能捕捉"连续低 anxiety 累加至高危"的模式。缺点是打破单向流动、破坏可审计性、与 Intervention 模块职责重叠。不选。跨轮趋势留给 Intervention 层处理。

### 异步 Router
将 Router 改为 async 以支持 LLM 降级时的高并发。优点是 LLM 等待时不阻塞线程。缺点是管线全线需要 async 改造，收益与投入不成比例——LLM 降级触发概率低，项目没有高并发需求。不选。
