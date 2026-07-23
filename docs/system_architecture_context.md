# 心理异常智能早筛与精准干预 — 代码架构上下文

本文档描述**调整后**的系统技术方案与**代码仓库目录/模块**的对应关系，供后续开发、评审与 Agent 上下文引用。图示来源：四阶段技术架构表 + Langchain 能力层 / 应用层架构图。

---

## 1. 总体目标

构建**多模态输入（音/视/文）→ 安全过滤 → 情感与结构标签 → 智能路由 → 干预闭环（通用 / 知识 / 安抚 / 危机）**的端到端能力，对外以 **FastAPI** 暴露统一接口，内部以 **LangChain 风格链路与 Runnable** 组织核心逻辑，底层可对接 **向量库（Milvus/Chroma）**、**轻量/大模型（Qwen 系等）**及 **TTS**。

---

## 2. 逻辑分层（与图示对应）

### 2.1 Langchain 能力层（核心业务编排）

| 组件 | 职责 | 代码落点（现状 / 规划） |
|------|------|-------------------------|
| 客户端入口 | Web 或硬件客户端；音视频与文本上行 | `api/routes/multimodal.py`、`static/`、`run_server.py` |
| 安全过滤器 | 敏感词库、正则、分级（如 L1 紧急 / L2 记录） | **`modules/safety/`**（Mock/Stub，`MOCK_SAFETY`）；规划将词库与正则迁入并实现 |
| 情感分析 + 上下文增强 | 多模态情感、融合标签 | `multimodal/`（ASR、表情、音频情感、`emotion_fusion.py`） |
| 路由决策中心 | `route(emotion, intensity, risk)` → `general` / `knowledge` / `comfort` / `crisis` | **`modules/router/`**（`MOCK_ROUTER`）；与 `TherapyChain` 对齐或拆分 |
| 情绪安抚链 | LLM + CoT：共情 → 分析 → 建议 | `core/chain/therapy_chain.py`、`modules/intervention/generator.py` |
| 知识检索链 | v3.3 RAG 管线 | **✅ 已实现**：`core/rag/`（QR + Union + gte-rerank-v2），详见 [设计文档](RAG-v3.3-设计文档.md) |
| 危机干预链 | 脚本 + `emergency_notify()` 模拟 | **规划**：`core/crisis/`（脚本模板、`emergency_notify`）；与现有 `safety_alert` 打通 |
| 心理慰藉干预大模型（输出整理） | 结构化输出：`reply`、内部 CoT 轨迹等 | `TherapyResponse`、`TherapyPromptTemplates` 扩展字段 |
| 最终响应（可选音频） | Edge-TTS 等 | `multimodal/tts.py` |

### 2.2 应用层（基础设施）

| 组件 | 职责 | 代码落点 |
|------|------|----------|
| 身份认证 | Shiro / OAuth2 等（可与网关集成） | **通常**：独立网关或服务；本仓库可为 FastAPI `Depends` 预留 |
| 会话与持久化 | session、对话与情绪记录 | `core/memory/session_memory.py`、`core/memory/db_storage.py`、`schemas/` |
| 向量数据库 | Chroma（本地持久化） | **✅ 已实现**：`core/rag/chroma_store.py`（Milvus 方案已废弃） |
| API 网关 | 路由、限流、鉴权 | 部署侧（Nginx/Kong 等）；本仓库保持 REST 契约稳定 |

---

## 3. 四阶段流水线（与表格模块对应）

### 阶段 A：输入与安全过滤

- **词库与规则**：JSON/CSV 敏感词分级。
- **实现要点**：中日文兼容、简繁与变体（如拆字）；统一 `SafetyCheckResult`（level、matched、span）。
- **多模态**：音频经 Whisper/FunASR 转写后走文本过滤；视频抽帧（如 1fps）可走轻量 NSFW/暴力打分后再汇总。
- **API**：`POST /api/v1/modules/safety/check`（详见 [`docs/parallel_module_io_samples.md`](parallel_module_io_samples.md)）。
- **紧急动作**：L1 → 调用 `emergency_notify()`、写审计日志。

### 阶段 B：情感分析

- **音频**：SenseVoice Small 等 → 文本 + 情感标签 + 声学事件（可选）。
- **视频/面部**：BlendFER-Lite / OpenFace 等（与本仓库 `multimodal/emotion.py` 路径并存，可按性能替换）。
- **文本**：Qwen 分类或专用情感模型。
- **融合**：`get_emotion_tags(...)` → 统一 `{ Emotion, Intensity, Risk }`（或扩展字段）；现有 `fuse_emotions` 可作为数值融合层。
- **API**：`POST /api/v1/modules/emotion/analyze`；多模态场景下可与 `multimodal` 转写结果合并。

### 阶段 C：智能路由

- **规则示例**：Risk ≥ 0.7 → crisis；0.5–0.7 → comfort（中高风险先安抚）；0.1–0.5 → knowledge（低风险给知识科普）；< 0.1 → general（无风险闲聊）。
- **接口**：`get_route(tags) -> Literal["general","knowledge","comfort","crisis"]`。
- **可选**：不满足规则覆盖率时用 MLP/随机森林学习 `route`（离线训练，在线只做推理）。

### 阶段 D：干预闭环

- **安抚**：CoT Prompt（共情→分析→建议）。
- **知识**：v3.3 RAG 管线（QR → Union(Chroma+BM25) → gte-rerank-v2 → LLM），详见 [RAG-v3.3-设计文档](RAG-v3.3-设计文档.md)。（旧规划：`text2vec-large-chinese` embedding、Milvus/Chroma —— 已废弃）
- **危机**：3–5 套脚本 + 紧急通知；响应时间目标：危机建议在 2 秒内返回（脚本路径）、安抚约 3 秒内、知识约 5 秒内（依赖检索与模型）。
- **编排**：`intervention_dispatcher(route, context) -> structured_response`，可选 TTS。

---

## 4. 本仓库目录映射（便于落地）

```
mental-intervene/
├── api/
│   ├── main.py                 # FastAPI 入口、静态资源、全局中间件
│   └── routes/
│       ├── chat.py             # 文本咨询 / 治疗链（遗留整合路径）
│       ├── multimodal.py       # 多模态会话、ASR/TTS、情感管线 API
│       ├── parallel_modules.py # 四模块独立契约接口（并行开发）
│       └── pipeline.py         # POST /api/v1/pipeline/run 端到端
├── modules/                    # 按业务分包：safety / emotion / router / intervention（mock 与 stub）
├── pipeline/
│   └── orchestrator.py         # 四阶段编排，仅传递 JSON（dict）
├── schemas/
│   ├── contracts/              # v1 契约 + samples/*.json
│   ├── models.py
│   └── database.py
├── core/
│   ├── chain/
│   │   └── therapy_chain.py    # 治疗对话链（情绪分析、安全、生成）
│   ├── llm/
│   │   └── base.py             # LLM 适配器（Qwen 等）
│   └── memory/                 # 会话记忆、持久化
├── multimodal/
│   ├── asr.py / audio_emotion.py / emotion.py / tts.py
│   ├── emotion_fusion.py

├── config/
│   ├── settings.py
│   └── prompts.py
├── utils/
└── docs/
    ├── system_architecture_context.md   # 本文档
    └── parallel_module_io_samples.md   # 各模块 HTTP 与 JSON 样例索引
```

**契约与并行开发**：跨模块边界以 `schemas/contracts/v1.py` 为准；HTTP 联调见 `docs/parallel_module_io_samples.md`，样例 JSON 在 `schemas/contracts/samples/`。

**仍可按图示新增**：

- `core/rag/` — 文档加载、切分、embedding、向量库、检索链（知识路由落地）。
- `core/crisis/` — 危机脚本与 `emergency_notify()`（与干预 Stub 合并或独立）。

---

## 5. HTTP 接口规划（与现有路由衔接）

| 端点 | 说明 | 备注 |
|------|------|------|
| `POST /api/v1/modules/safety/check` | 安全过滤 | 见 `parallel_modules` |
| `POST /api/v1/modules/emotion/analyze` | 情感分析 | 见 `parallel_modules` |
| `POST /api/v1/modules/router/route` | 智能路由 | 见 `parallel_modules` |
| `POST /api/v1/modules/intervention/run` | 干预闭环 | 见 `parallel_modules` |
| `POST /api/v1/pipeline/run` | 四阶段串联 | 见 `pipeline` |
| `POST /api/v1/multimodal/...` | 多模态会话 | 已有 |
| `POST /api/v1/chat` 等 | 治疗链对话 | 已有（遗留路径） |

版本前缀保持 `/api/v1/` 一致，便于网关与前端聚合。

---

## 6. 核心数据契约（建议）

统一在 `schemas/` 中定义，供 API 与链路透传：

- **`EmotionTags`**：`primary_emotion`、`intensity`（数值或枚举）、`risk`（0–1）、`modality_breakdown`（可选）。
- **`SafetyCheckResult`**：`level`、`matched_terms`、`action`（pass / block / escalate）。
- **`RouteDecision`**：`route`、`reason`、`confidence`（可选）。
- **`InterventionOutput`**：`text`、`empathy`、`suggestion`、`action_items`、`chain_of_thought`（对内）、`tts_audio_base64`（可选）。

---

## 7. 依赖与环境（与方案表一致）

- **运行时**：Python 3.8+，FastAPI，LangChain 核心（`langchain_core` 等）。
- **模型与媒体**：PyTorch、Transformers；音视频侧 Librosa、OpenCV、ffmpeg（进程外）；ASR（Whisper / FunASR）；TTS（Edge-TTS 等）。
- **向量检索**：Chroma（bge-m3 1024-dim，ollama 本地）+ BM25 关键词（jieba 分词）。

---

## 8. 演进顺序（实施建议）

1. **在 `modules/safety` 落地真实词库与正则**，与 `TherapyChain` 安全检测对齐或改为调用统一契约。
2. **完善 `modules/router` 规则或 ML**，与 `TherapyChain` 分支策略对齐。
3. **RAG 已接入**（`core/rag` v3.3），知识链与安抚链共用同一 LLM 适配器。
4. **危机链产品化**：脚本库、`emergency_notify`、审计与 SLA 监控。
5. **指标**：召回率、误报率、各模态准确率与端到端延迟压测。

---

## 9. 文档维护

方案或目录有重大变更时，同步更新本节与第 4 节映射表；新增对外 HTTP 契约时更新第 5、6 节。

*文档版本：与「心理异常智能早筛与精准干预」调整方案对齐，生成日期以仓库变更记录为准。*
