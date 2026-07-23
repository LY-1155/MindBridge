# 并行开发：各模块 HTTP 路径与 JSON 样例

契约定义见 `schemas/contracts/v1.py`；同名样例文件位于 `schemas/contracts/samples/`（可直接 `curl -H "Content-Type: application/json" -d @file` 联调）。

默认 `MOCK_SAFETY` / `MOCK_EMOTION` / `MOCK_ROUTER` / `MOCK_INTERVENTION` 均为 `true`（见 `config/settings.py`），响应为 Mock 固定形态。将对应项设为 `false` 且重启服务后，会使用 Stub（`modules/*/stub.py`），便于逐步替换真实实现。

---

## 统一约定

| 字段 | 说明 |
|------|------|
| `contract_version` | 当前为 `1.0`，破坏性变更时升级并评审 |

**Base URL**：与主服务一致，例如 `http://localhost:8000`。

---

## 1. 输入与安全过滤

| | |
|---|---|
| **HTTP** | `POST /api/v1/modules/safety/check` |
| **请求模型** | `SafetyCheckRequest` |
| **响应模型** | `SafetyCheckResult` |
| **样例文件** | `samples/safety_request.json` → `samples/safety_response.json` |

**请求要点**：`text` 为待检测正文（可由上游 ASR 合并）；`blocked=true` 或 `level ≥ 2`（见 `pipeline/orchestrator.py` 中 `EMERGENCY_SAFETY_LEVEL`）时，端到端流水线将短路危机路径。

---

## 2. 情感分析

| | |
|---|---|
| **HTTP** | `POST /api/v1/modules/emotion/analyze` |
| **请求模型** | `EmotionAnalyzeRequest` |
| **响应模型** | `EmotionTags` |
| **样例文件** | `samples/emotion_request.json` → `samples/emotion_response.json` |

**请求要点**：`safety` 必须为上一阶段 **`SafetyCheckResult.model_dump()`** 的可 JSON 序列化字典。

---

## 3. 智能路由

| | |
|---|---|
| **HTTP** | `POST /api/v1/modules/router/route` |
| **请求模型** | `RouteRequest` |
| **响应模型** | `RouteDecision` |
| **样例文件** | `samples/router_request.json` → `samples/router_response.json` |

**请求要点**：`emotion`、`safety` 分别为 **`EmotionTags`** 与 **`SafetyCheckResult`** 的 dict。  
**路由规则**（`MOCK_ROUTER=false`）：`risk ≥ 0.7` → `crisis`；`0.5 ≤ risk < 0.7` → `comfort`（中高风险先安抚）；`0.1 ≤ risk < 0.5` → `knowledge`（低风险给知识科普）；`risk < 0.1` → `general`（无风险闲聊）。Mock 下恒为 `comfort / mock_default`。

---

## 4. 干预闭环

| | |
|---|---|
| **HTTP** | `POST /api/v1/modules/intervention/run` |
| **请求模型** | `InterventionRequest` |
| **响应模型** | `InterventionResult` |
| **样例文件** | `samples/intervention_request.json` → `samples/intervention_response.json` |

**请求要点**：`route`、`emotion`、`safety` 均为上游模块输出的 dict；`user_text` 为用户原始文本。

**深入说明（需求完成度、安全/情感/路由如何喂给干预、代码与 HTTP 样例）**：见 [`module_intervention_integration.md`](module_intervention_integration.md)。

---

## 5. 端到端流水线（串联四模块）

| | |
|---|---|
| **HTTP** | `POST /api/v1/pipeline/run` |
| **请求模型** | `PipelineInput` |
| **响应模型** | `PipelineOutput` |
| **样例文件** | `samples/pipeline_request.json` → `samples/pipeline_response.json` |

**响应要点**：`stopped_after_safety=true` 表示安全短路，此时 `emotion` / `route` 为编排层占位，仍以 JSON 填满便于前端统一解析。

---

## 6. 与遗留能力的关系

| 路径 | 说明 |
|------|------|
| `POST /api/v1/chat` | 既有 **治疗对话链**（`core/chain/therapy_chain.py`），未改为契约流水线 |
| `POST /api/v1/multimodal/...` | 多模态预处理；可将转写文本与 `session_id` 传入 `PipelineInput.text` 做并行方案集成 |

后续可在 `modules/intervention/` 内增加适配器，将 `InterventionRequest` 委托给 `TherapyChain`，再逐步收敛为单一干预实现。
