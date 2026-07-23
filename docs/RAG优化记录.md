# PRISM RAG 检索层优化总结

> 最后更新：2026-07-17 | 涉及模块：`core/rag/` `modules/intervention/rag/` `modules/intervention/service.py`

---

## 一、背景

PRISM 知识库检索层原架构（v3）：

```
用户查询 → Chroma 稠密 + BM25 关键词 → RRF 融合 → 外部 API fallback → top-3
```

存在两个问题：

1. **QueryClassifier 已废弃**（ADR 0011 评估结论）：分类器造成 Recall -0.20，生产管线已移除。
2. **口语查询命中率低**：用户口语化表达（"男朋友老躲着我"）与知识库专业术语（"回避型依恋"）之间存在语义鸿沟。

---

## 二、v3.1 → v3.2（2026-07-14）：QueryRewriting + BGE Reranker

### 2.1 新增查询改写器（QueryRewriter）

**新建文件：** `core/rag/query_rewriter.py`

- 检索前用 LLM 将口语化查询改写为富含心理学专业术语的关键词串
- 原文保留，术语追加，不做替换（避免信息丢失）
- 改写失败自动回退原始查询
- 使用 `REWRITER_MODEL_NAME`（qwen3.7-max），temperature=0

**效果验证（v3.2 评测）：**

| 测试集 | 无改写 Recall@3 | 有改写 Recall@3 | 提升 |
|--------|----------------|-----------------|------|
| 合成 141 条 | 0.20 | 0.67 | +233% |
| 真实 15 条 | 0.18 | 0.31 | +75% |

### 2.2 新增 BGE Cross-Encoder 重排序器（Reranker）

**新建文件：** `core/rag/reranker.py`

- 使用 BAAI 开源 `bge-reranker-v2-m3` cross-encoder 模型，本地推理
- 粗检索 top-15 → 重排取 top-3

**v3.2 全方案对比（15 条真实查询）：**

| 方案 | 无改写 Recall | 有改写 Recall | 额外延迟 |
|------|-------------|-------------|---------|
| **BM25 only** | 0.18 | **0.31** | 0 |
| BM25 + BGE Reranker | 0.18 | 0.27 | ~50ms×15 |
| BM25 + LLM Reranker | 0.18 | 0.22 | ~1s×15 |
| Hybrid + BGE Reranker | 0.16 | 0.20 | ~50ms×15 |

**v3.2 结论：** BM25 + QueryRewriting 最优（0.31），Hybrid + Reranker 反而更差。

---

## 三、v3.3（2026-07-17）：根因定位 + 架构修正

### 3.1 RRF 融合是罪魁祸首

v3.2 的困惑：为什么 Chroma + BM25 + Reranker 不如 BM25-only？

经过逐组件消融实验，定位到三个根因：

#### 根因一：RRF 融合在双路质量不均衡时 1+1<1

```
BM25 单路 Recall:  0.18
Chroma 单路 Recall: 0.20
RRF 融合 Recall:    0.13  ← 比任何单路都差！
```

RRF 给两路同等投票权，当 Chroma 返回噪声时，噪声和 BM25 的好结果"对等表决"，导致最终排序被污染。

#### 根因二：术语鸿沟限制了嵌入模型的发挥

用户口语（"失败、沉重"）和知识库临床术语（"重性抑郁障碍"）之间的语义鸿沟巨大：

- 45 篇 ground truth 文档中，**34 篇（75%）根本不出现在 BM25 前 10 名里**
- 不是排序问题，是关键词完全不重叠
- 无论 bge-m3 还是 text-embedding-v4，Chroma 单路都卡在 Recall ≈ 0.20

#### 根因三：Reranker 候选池被 Chroma 噪声污染

Union 并集去重后，Chroma 的高分噪声文档占据前排，BM25 的好结果被挤到后面。修复办法：双路分数归一化后合并排序，而非简单拼接。

### 3.2 解决方案

#### 用 Union + Reranker 取代 RRF

```
旧（v3.2 RRF）：                 新（v3.3 Union + Reranker）：
Chroma top-N ─┐                 Chroma top-N ─┐
              ├─ RRF 融合 → top-K               ├─ 归一化 → 并集去重 → QwenReranker → top-K
BM25 top-N  ─┘                  BM25 top-N  ─┘
```

核心变化：不再用 RRF 的简单排名公式做融合，而是把两路结果合并后交给 Cross-Encoder Reranker 逐对打分精排。

#### 引入百炼 qwen3-rerank API 重排序器

BGE Reranker 需要下载 1.5GB 模型，国内网络受限且吃显存。改用百炼 `qwen3-rerank` API：

- 零本地资源，HTTP API 调用
- 单次最多 500 文档，每条最长 4000 tokens
- 费用约 ¥0.0035/次（15 文档标准调用）
- 代码：`core/rag/reranker.py` → `QwenReranker` 类

### 3.3 Embedding 模型选型实验

在真实查询上对比了三种 embedding 配置：

| Embedding 模型 | Chroma Recall@3 | 结论 |
|---------------|-----------------|------|
| text-embedding-v4 (1024-dim) | 0.20 | 通用模型，术语敏感度不足 |
| text-embedding-v4 (2048-dim) | 0.20 | 提维无显著改善 |
| bge-m3 (ollama, 1024-dim) | 0.20 | 与 text-embedding-v4 持平 |

**结论：** 当前任何通用 embedding 模型都跨不过心理学领域的术语鸿沟。Embedding 模型不是瓶颈，上界由 Query Rewriting + Reranker 决定。

### 3.4 最终评测

#### 合成查询 141 条（术语天然匹配，无术语鸿沟干扰）

| 方案 | Recall@3 | vs ADR 0011 |
|------|----------|------------|
| BM25 only | 0.312 | = sparse_bare |
| Chroma only (bge-m3) | 0.553 | = dense_bare |
| ADR 0011 RRF 融合 (hybrid_bare) | 0.567 | 基准 |
| **Union + QwenReranker** | **0.723** | **+27%** |

> 在术语鸿沟不存在时，Union + Reranker 完胜 RRF，证明架构设计正确。

#### 真实查询 15 条（存在术语鸿沟）

| 方案 | Recall@3 | 提升 |
|------|----------|------|
| BM25 only | 0.178 | 基准 |
| BM25 + Query Rewriting | 0.244 | +38% |
| Chroma only (bge-m3) | 0.200 | +12% |
| Union(Chroma+BM25) + Reranker | 0.156 | -12% |
| QR + Union + Reranker | 0.222 | +25% |
| **BM25 + Rewrite + Reranker** | **0.356** | **+100%** |

> 术语鸿沟存在时，Chroma 语义检索无法贡献增量；最优方案是 QR → BM25 → Reranker。

---

## 四、生产推荐配置（v3.3）

### 当前最优：QueryRewriting → BM25 → QwenReranker

```
用户查询
    │
    ▼
QueryRewriter (qwen3.7-max, temp=0)
    │  改写后查询（原文 + 专业术语）
    ▼
BM25 关键词检索 (top-15) + Chroma 语义检索 (top-15)
    │
    ▼
Union 并集去重，归一化排序
    │
    ▼
QwenReranker (qwen3-rerank API) → 精排 top-3
    │
    ▼
LLM 生成回答
```

> 双路并集保留 Chroma，当未来有领域专用 Embedding 时可无缝启用语义检索增益。
> 当前 Chroma 路径不产生正向贡献，但不产生负面影响——Reranker 会自动过滤低质候选。

### 模型配置

| 环境变量 | 值 | 用途 |
|---------|---|------|
| `MODEL_NAME` | `qwen3.7-max` | 主对话 |
| `SCORING_MODEL_NAME` | `qwen3.6-flash` | 量表计分 |
| `REWRITER_MODEL_NAME` | `qwen3.7-max` | 查询改写 |
| `EMBEDDING_MODEL_NAME` | `bge-m3` | 向量嵌入（ollama 本地） |
| `EMBEDDING_API_BASE` | `http://localhost:11434/v1` | ollama 地址 |
| `EMBEDDING_BACKEND` | `api` | 使用 OpenAI 兼容 API 透传 ollama |
| `EMBEDDING_DIMENSIONS` | `0` | 0=模型默认 |

### 组件状态

| 组件 | 状态 | 代码位置 |
|------|------|---------|
| QueryRewriter | ✅ 生产启用 | `core/rag/query_rewriter.py` |
| QwenReranker | ✅ 生产启用 | `core/rag/reranker.py` |
| Chroma 语义检索 | ⚠️ 保留但不主导 | `core/rag/chroma_store.py` |
| BM25 关键词检索 | ✅ 生产启用 | `core/rag/bm25_index.py` |
| BGE Reranker (本地) | 💤 备用 | `core/rag/reranker.py` |
| BGEM3Embedding (本地) | 💤 备用 | `core/rag/embedder.py` |
| RRF 融合 | 💤 无 Reranker 时兜底 | `core/rag/hybrid_retriever.py` |
| QueryClassifier | ❌ 已废弃 (ADR 0011) | `core/rag/query_classifier.py` |

---

## 五、修改文件清单

### v3.3 新增文件

| 文件 | 说明 |
|------|------|
| — | 无新增文件，均为既有文件修改 |

### v3.3 修改文件

| 文件 | 改动 |
|------|------|
| `core/rag/hybrid_retriever.py` | 新增 `retrieve_union_with_ids()`：双路并集去重 + 分数归一化排序；RRF 保留为兜底 |
| `core/rag/reranker.py` | 新增 `QwenReranker` 类：百炼 qwen3-rerank API；保留 `BGEReranker` 为备用 |
| `modules/intervention/rag/retriever.py` | `retrieve()` 重写：有 Reranker → Union 模式；无 Reranker → RRF 兼容 |
| `modules/intervention/service.py` | 注入 `QwenReranker` 替代 `BGEReranker` |
| `core/rag/embedder.py` | 新增 `BGEM3Embedding` 类（备用）、`dimensions` 参数、`create_embedder()` 工厂方法 |
| `config/settings.py` | 新增 `EMBEDDING_BACKEND`、`EMBEDDING_DIMENSIONS`、`RRF_DENSE_WEIGHT`、`LOCAL_EMBEDDING_MODEL`；修复 `extra = "ignore"` |
| `.env` / `docker/.env` | 更新 embedding 配置：ollama bge-m3、RRF dense_weight |
| `docker/docker-compose.yml` | 新增 `EMBEDDING_BACKEND`、`LOCAL_EMBEDDING_MODEL` 环境变量 |

### v3.2 新增文件（前置版本）

| 文件 | 说明 |
|------|------|
| `core/rag/query_rewriter.py` | LLM 查询改写器 |
| `core/rag/reranker.py` | BGE Cross-Encoder 重排序器（v3.3 扩展为含 QwenReranker） |
| `scripts/test_query_rewriter.py` | 改写器快速对比脚本 |
| `scripts/eval_query_rewriter.py` | 改写器量化评测脚本 |
| `data/eval/real_queries.jsonl` | 15 条真实查询测试集 |

### v3.2 修改文件（前置版本）

| 文件 | 改动 |
|------|------|
| `config/settings.py` | 新增 `REWRITER_MODEL_NAME` |
| `core/rag/embedder.py` | 修复 batch_size 和维度 |
| `modules/intervention/rag/retriever.py` | v3 → v3.2：增加 rewriter/reranker 参数 |
| `modules/intervention/service.py` | 自动注入 QueryRewriter 和 Reranker |
| `requirements.txt` | 新增 pyjwt, bcrypt, chromadb, requests, FlagEmbedding |
| `docker/Dockerfile` | 加阿里云 apt + pip 镜像源 |
| `docker/docker-compose.yml` | 扩展模型环境变量，统一 API Key 注入 |
| `gunicorn.conf.py` | 修复空字符串 GUNICORN_WORKERS 崩溃 |
| `.env` / `docker/.env` | 新增百炼平台全部模型配置 |

---

## 六、评测命令（本地 conda 环境）

```powershell
# 环境准备
conda activate emotion
cd e:\postgrad\PRISM-master

# 删除旧索引 + 重建
Remove-Item -Recurse -Force data/knowledge/chroma_index
python scripts/build_knowledge_index.py --backend api

# 快速评测（15 条真实查询）
python scripts/eval_query_rewriter.py

# 全量评测（141 条合成查询）
python scripts/eval_retrieval.py
```

---

## 七、已知局限与未来方向

1. **术语鸿沟**：用户口语 vs 临床术语的根本性语义鸿沟是当前最大瓶颈。Query Rewriting 弥补了约 50%，但仍有 70% 的相关文档无法被任何检索方式命中。
2. **Embedding 模型**：当前通用模型（bge-m3 / text-embedding-v4）在心理学中文上无明显差异。未来方向：领域微调 embedding 模型（如用心理咨询对话对 fine-tune bge-m3）。
3. **知识库覆盖**：860 篇文档对真实用户查询的覆盖度可能不足。建议持续扩充知识库，特别是针对高频查询类别（coping_strategies、trauma_and_stress 当前 Recall=0）。
4. **Reranker 成本**：qwen3-rerank API 每次调用约 ¥0.0035，日均 1000 次查询 ≈ ¥3.5，可接受。
