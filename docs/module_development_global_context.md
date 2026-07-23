# 四模块实现细化与全局开发上下文（V1）

本文档用于统一后续四模块开发需求的默认上下文。凡涉及 `modules/`、`pipeline/`、`api/routes/parallel_modules.py`、`api/routes/pipeline.py`、`schemas/contracts/` 的需求，均以本文档为基线。

---

## 1. 目标与范围

- 构建稳定的四阶段闭环：`安全过滤 -> 情感分析 -> 智能路由 -> 干预闭环`。
- 保持模块解耦：模块间仅通过契约 JSON 交互，不依赖彼此内部实现。
- 在保持 `contract_version=1.0` 兼容前提下，逐步从 Mock/Stub 替换到真实实现。

---

## 2. 全局约束（所有模块共同遵守）

### 2.1 契约与接口约束

- 模块间数据传递统一使用 `model_dump()` 后的 dict。
- 现有字段语义不得破坏；新增字段默认可选并提供默认值。
- 对外接口保持稳定：
  - `POST /api/v1/modules/safety/check`
  - `POST /api/v1/modules/emotion/analyze`
  - `POST /api/v1/modules/router/route`
  - `POST /api/v1/modules/intervention/run`
  - `POST /api/v1/pipeline/run`

### 2.2 编排与降级约束

- 统一编排入口为 `pipeline/orchestrator.py`。
- 安全短路规则：`blocked=true` 或 `level >= 2` 时，情感与路由走占位输出，但干预阶段仍执行。
- 任一模块失败时，必须给出可追踪原因（`reason/meta`）并保留最小可用响应形状。

### 2.3 性能与可观测性目标（阶段性）

- 危机场景（脚本路径）目标响应：`<= 2s`
- 安抚场景目标响应：`<= 3s`
- 知识场景目标响应：`<= 5s`
- 所有模块输出应携带可审计信息（至少包含关键决策原因和必要 meta）。

---

## 3. 模块 A：输入与安全过滤（Safety）

### 3.1 输入输出

- 输入：`SafetyCheckRequest`（核心字段：`text`, `session_id`, `contract_version`）
- 输出：`SafetyCheckResult`（核心字段：`level`, `blocked`, `matched_terms`, `meta`）

### 3.2 实现内容细化

1. 文本归一化层：
   - 中英混输标准化、全半角处理、大小写归一、基础噪声清理。
   - 敏感词变体处理（同音/拆字/常见替换）以提高召回。
2. 规则引擎层：
   - 词库规则（JSON/CSV）+ 正则规则双通道。
   - 输出命中项与规则来源，便于审计与调参。
3. 风险分级与动作映射：
   - `level=0`: pass
   - `level=1`: warn/record
   - `level>=2`: block/escalate（触发危机链）
4. 多模态接入策略：
   - 语音文本来自 ASR 转写后进入同一安全检查。
   - 视频风险信号可先汇总为文本标签再并入规则判断（后续扩展）。
5. 联调与测试：
   - 至少覆盖：普通文本、弱风险文本、高风险文本、空文本/异常输入。

### 3.3 验收标准

- 高风险样本必须稳定输出 `blocked=true` 或 `level>=2`。
- `matched_terms` 与 `meta` 可解释，不允许仅返回黑盒分值。

---

## 4. 模块 B：情感分析（Emotion）

### 4.1 输入输出

- 输入：`EmotionAnalyzeRequest`（`text`, `safety`, `session_id`）
- 输出：`EmotionTags`（`primary_emotion`, `intensity`, `risk`, `modality_notes`）

### 4.2 实现内容细化

1. 单模态能力：
   - 文本情感识别（基础分类 + 强度估计）。
   - 可选音频/视频情感特征通道（由多模态模块提供）。
2. 融合策略：
   - 先给出可解释的规则/加权融合基线，再逐步替换为学习模型。
   - `risk` 统一归一到 `[0,1]`。
3. 与安全模块协同：
   - 当上游安全等级升高时，情感模块可提升风险保守性（如最小风险下限）。
4. 失败降级：
   - 任一子通道失败时保留主输出结构，`modality_notes` 标注缺失来源。
5. 联调与测试：
   - 至少覆盖：中性、负向、高压、异常输入、上游 safety 高等级输入。

### 4.3 验收标准

- 输出字段完整且数值范围合法（`intensity/risk` 在预设范围内）。
- `primary_emotion` 与 `risk` 在典型样本上方向正确（可用规则用例验收）。

---

## 5. 模块 C：智能路由（Router）

### 5.1 输入输出

- 输入：`RouteRequest`（`emotion`, `safety`）
- 输出：`RouteDecision`（`route`, `reason`, `confidence`）

### 5.2 实现内容细化

1. 规则基线（默认可落地）：
   - `risk >= 0.7 -> crisis`
   - `0.3 <= risk < 0.7 -> knowledge`
   - `risk < 0.3 -> comfort`
2. 安全优先策略：
   - 若安全模块给出紧急标志，则强制 `crisis`（覆盖情感低风险判断）。
3. 可解释性：
   - `reason` 必须说明关键触发条件（如阈值命中、规则编号）。
4. 置信度策略：
   - 规则路由给出稳定区间置信度；学习模型接入后可替换为概率输出。
5. 联调与测试：
   - 三类路由都需有固定测试样本；边界值（0.29/0.30/0.69/0.70）必须覆盖。

### 5.3 验收标准

- 路由结果可解释且稳定，不允许同输入随机漂移。
- 危机优先级高于一般路由逻辑。

---

## 6. 模块 D：干预闭环（Intervention）

### 6.1 输入输出

- 输入：`InterventionRequest`（`user_text`, `route`, `emotion`, `safety`, `session_id`）
- 输出：`InterventionResult`（`reply`, `strategy`, `emergency_triggered`, `meta`）

### 6.2 实现内容细化

1. `comfort`（安抚链）：
   - 目标：共情确认、情绪命名、短步建议、可执行行动项。
   - 可复用 `TherapyChain` 或专用提示词链路。
2. `knowledge`（知识链）：
   - 目标：基于检索结果给出心理教育/自助策略。
   - 预留 RAG 接入点：检索、重排、回答生成。
3. `crisis`（危机链）：
   - 目标：优先保障安全，使用固定脚本快速响应。
   - 必须包含：危机提示语 + `emergency_notify()`（可先日志模拟） + 审计记录。
4. 结构化输出：
   - `reply` 面向用户，`strategy/meta` 面向系统与运营。
   - `emergency_triggered` 与路由结果一致，不允许漏标。
5. 联调与测试：
   - `general/knowledge/comfort/crisis` 四路由样例全覆盖。
   - 验证危机路径通知函数被触发（可通过 mock 断言）。

### 6.3 验收标准

- 三分支可独立执行且输出结构一致。
- 危机分支优先级最高，通知动作可观测可追踪。

---

## 7. 端到端编排细化

1. 串联顺序固定：`Safety -> Emotion -> Router -> Intervention`。
2. 短路策略固定：安全紧急时跳过真实 Emotion/Router 调用并填充占位 JSON。
3. 响应结构固定：`PipelineOutput` 必须包含四模块结果与 `stopped_after_safety`。
4. 编排层只依赖 `modules/ports.py` 协议，具体实现由 `modules/factory.py` 装配。

---

## 8. 开发完成定义（DoD）

每个模块开发任务完成前，至少满足：

1. 契约兼容：不破坏现有字段语义。
2. 实现可替换：通过工厂装配点切换 Mock/Stub/Real。
3. 测试通过：模块接口测试 + 编排测试至少各 1 条。
4. 文档同步：更新样例 JSON 或模块文档说明。
5. 联调可用：`/api/v1/pipeline/run` 冒烟通过。

---

## 9. 关联文档

- 架构基线：`docs/system_architecture_context.md`
- 协同流程：`docs/team_collaboration_playbook.md`
- 模块 IO 样例：`docs/parallel_module_io_samples.md`
- 干预专项说明：`docs/module_intervention_integration.md`
- 提单模板：`docs/cursor_module_requirement_template.md`

---

## 10. 使用说明（重要）

- 本文档是后续 Cursor 开发需求的默认全局上下文。
- 当需求与本文档冲突时，以用户在当次会话的明确指令为准，并在实现说明中标注偏离点。
