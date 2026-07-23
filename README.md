# PRISM — 心理异常智能早筛与精准干预系统

基于大语言模型（LLM）的多模态心理危机早筛与四级路由干预闭环系统。支持文本、语音、视频三路输入，通过 LangGraph 编排安全过滤 → 情绪分析 → 智能路由 → 干预闭环四阶段管线，按路由分支（通用/知识/安抚/危机，risk 从低到高）生成定制化 AI 回复。

## 架构概览

```
用户输入（文本 / 音频 / 视频）
        │
        ▼
┌──────────────────────────────────────────────────┐
│  视频预处理（ffmpeg 分离音视频 → 抽帧 + 人脸情绪）   │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  LangGraph StateGraph 四阶段管线                       │
│                                                      │
│  ① Safety ──→ ② Emotion ──→ ③ Router ──→ ④ Intervention │
│       │                                        ▲       │
│       └─── 安全短路（level ≥ 2）──→ crisis ─────┘       │
│                                                      │
│  Route:  general / knowledge / comfort / crisis       │
└──────────────────────────────────────────────────────┘
        │
        ▼
   结构化 AI 回复（共情 / 知识 / 量表筛查 / 危机话术）
```

## 核心特性

### 多模态情绪分析
- **文本情绪**：策略模式双引擎 —— ONNX 语义模型 + 关键词引擎，自动降级
- **语音情绪**：SenseVoice 单次推理同时产出 ASR 转写 + 情绪标签
- **视觉情绪**：MediaPipe 人脸检测 → HSEmotion 推理（Ekman 7 类）→ 8 类契约标签映射
- **情绪融合**：三路置信度加权聚合 + 信号冲突仲裁 + 时序加权

### 四级智能路由
| 路由 | 触发条件 | 干预策略 |
|------|---------|---------|
| **General** | 极低风险，信息提问或日常闲聊 | LLM 自然对话，不强行转向心理话题 |
| **Comfort** | 低-中等风险，情感表达 | LLM 共情倾听，不给建议 |
| **Knowledge** | 中等风险，需要认知干预 | RAG 知识检索 + LLM 生成 + 量表筛查 |
| **Crisis** | 高风险或安全短路 | 确定性话术模板（非 LLM 生成），紧急热线推送 |

### 知识库 RAG
- **查询分类**：LLM 将用户查询分为 9 类心理学知识域 → Chroma metadata 过滤
- **混合检索**：稠密向量（百炼 Embedding）+ jieba-BM25 关键词 → RRF 融合排序
- **实时兜底**：Tavily Search API 补充 60+ 种知识库未覆盖的精神科药物信息

### 量表筛查
- 支持 PHQ-9（抑郁）、GAD-7（焦虑）标准化量表
- 多轮自然对话式引导（LLM 自由组织句式，不逐字念题）
- PydanticOutputParser 结构化计分（0-3 锚点，-1 话题偏离自动拒绝）
- D10 串行多量表连续执行，结果反哺 RAG 检索

### 安全与治理
- **安全过滤**：敏感词分级（0/1/2），level ≥ 2 触发管线短路
- **安全标记累积**：滑动窗口内 level=1 累计达阈值自动升段至 crisis
- **紧急推送**：4 类危机话术模板 + 会话级冷却防重复 + webhook 告警（钉钉/飞书）
- **字段级加密**：AES-256-GCM 加密敏感字段（消息内容、情绪上下文、安全标记）
- **速率限制**：slowapi + Redis，60 req/min/user，危机端点不限流
- **Prompt 防护**：instruction hierarchy + 用户输入包裹防注入

### 用户体系
- JWT 双 token 认证（30min access + 30d refresh，服务端可吊销）
- bcrypt 密码哈希，credential 与 user 解耦（支持多种登录方式扩展）
- 账号软删除（30 天后悔期）+ 定时物理清理
- 用户数据导出（全部会话/情绪/量表/安全标记，JSON 格式）
- 知情同意弹窗 + AI 辅助标注

## 技术栈

| 层级 | 技术 |
|------|------|
| **Web 框架** | FastAPI + Gunicorn/Uvicorn |
| **LLM 编排** | LangGraph (StateGraph) + LangChain + LCEL |
| **数据库** | MySQL + SQLAlchemy + Alembic (migrations) |
| **缓存/限流** | Redis + slowapi |
| **向量检索** | Chroma + 百炼 Embedding API + jieba-BM25 |
| **模型推理** | ONNX Runtime / SenseVoice / HSEmotion / MediaPipe |
| **安全** | AES-256-GCM / bcrypt / JWT |
| **部署** | Docker Compose (FastAPI + Nginx + Redis + MySQL) |
| **日志** | 结构化 logging（request_id/user_id/session_id 链路追踪） |

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+
- Redis 7+
- ffmpeg（视频预处理）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/xuteng-412/PRISM.git
cd PRISM

# 2. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY、数据库密码等

# 5. 执行数据库迁移
alembic upgrade head

# 6. 启动服务
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看 Swagger API 文档。

### Docker 部署

```bash
# 配置 .env 后一键启动
docker compose -f docker/docker-compose.yml --env-file .env up -d

# 查看服务状态
docker compose -f docker/docker-compose.yml ps
```

## API 概览

| 端点 | 说明 |
|------|------|
| `POST /api/v1/pipeline/run` | 端到端四阶段管线（文本/音频） |
| `POST /api/v1/pipeline/run/video` | 端到端管线（视频输入，含预处理） |
| `POST /api/v1/modules/safety/check` | 独立安全过滤 |
| `POST /api/v1/modules/emotion/analyze` | 独立情绪分析 |
| `POST /api/v1/modules/router/route` | 独立路由决策 |
| `POST /api/v1/modules/intervention/run` | 独立干预生成 |
| `POST /api/v1/chat` | 文本咨询（遗留治疗链路径） |
| `POST /api/v1/auth/register` | 用户注册 |
| `POST /api/v1/auth/login` | 用户登录 |
| `GET /api/v1/user/export` | 用户数据导出 |
| `GET /api/v1/health` | 健康检查 |
| `GET /ping` | 存活探测 |

## 项目结构

```
PRISM/
├── api/                        # FastAPI 入口 + 路由
│   ├── main.py                 # 应用入口，生命周期管理
│   └── routes/                 # auth / chat / multimodal / pipeline / user
├── pipeline/                   # 管线编排
│   └── orchestrator.py         # LangGraph StateGraph 四阶段编排
├── modules/                    # 业务模块
│   ├── safety/                 # 安全过滤 + 紧急推送 + 视频安全
│   ├── emotion/                # 情绪分析（ONNX/关键词双引擎 + 多模态融合）
│   ├── router/                 # 智能路由（risk bands + emotion bias）
│   ├── intervention/           # 干预闭环（generator / crisis / scale / RAG）
│   ├── auth_service.py         # 认证服务（bcrypt + credential）
│   ├── user_service.py         # 用户服务（软删除 + 数据导出）
│   ├── encryption.py           # AES-256-GCM 字段加密
│   ├── rate_limit.py           # slowapi + Redis 限流
│   └── prompt_guard.py         # Prompt 注入防护
├── core/                       # 核心基础设施
│   ├── llm/                    # LLM 适配器（Qwen 等）
│   ├── memory/                 # 会话记忆（Redis + MySQL）
│   ├── rag/                    # RAG（query_classifier / search_fallback）
│   └── chain/                  # 治疗对话链（遗留路径）
├── config/                     # 配置
│   ├── settings.py             # Pydantic Settings
│   ├── router_rules.json       # 路由规则配置
│   └── logging_config.py       # 结构化日志配置
├── schemas/                    # 数据模型
│   ├── contracts/v1.py         # 跨模块 JSON 契约（v1.4）
│   ├── database.py             # SQLAlchemy 模型
│   └── database_v2.py          # 用户体系扩展模型
├── migrations/                 # Alembic 数据库迁移
├── docker/                     # Docker 配置
│   ├── Dockerfile
│   ├── docker-compose.yml      # 生产环境
│   └── docker-entrypoint.sh
├── nginx/                      # Nginx 反向代理 + SSL
├── static/                     # 前端静态资源
├── tests/                      # 测试用例（62 个测试文件）
├── docs/                       # 文档
│   ├── system_architecture_context.md
│   ├── runbook.md              # 运维操作手册
│   └── backup-strategy.md      # 备份策略
├── data/knowledge/             # 知识库（9 类心理学知识）
└── scripts/                    # 工具脚本
```

## 配置参考

关键环境变量（详见 `.env.example`）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MODEL_NAME` | 主对话模型 | `qwen2.5-7b-instruct` |
| `SCORING_MODEL_NAME` | 量表计分模型（轻量） | `qwen-turbo` |
| `EMOTION_ENGINE` | 文本情绪引擎 | `keyword`（可选 `onnx`） |
| `ASR_BACKEND` | 语音识别后端 | `sensevoice` |
| `ENCRYPTION_KEY` | AES-256 加密密钥 | 生产必须更换 |
| `JWT_SECRET_KEY` | JWT 签名密钥 | 生产必须更换 |

## License

MIT
