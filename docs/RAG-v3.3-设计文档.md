# PRISM RAG 检索层 v3.3 设计文档

> 最后更新：2026-07-17 | 实验记录详见 [RAG优化记录.md](RAG优化记录.md)

---

## 一、整体架构

```
用户输入
    │
    ▼
┌─────────────────────────────────────┐
│  Router（路由判断）                   │
│  crisis / comfort / general → 不走RAG │
│  knowledge → 走RAG                   │
└─────────────────────────────────────┘
    │ knowledge
    ▼
┌─────────────────────────────────────┐
│  Step 1: Query Rewriting             │
│  LLM (qwen3.7-max, temp=0)           │
│  口语 → 心理学术语关键词              │
│  原文保留 + 术语追加，不做替换         │
└─────────────────────────────────────┘
    │ 改写后的查询
    ▼
┌─────────────────────────────────────┐
│  Step 2: 双路粗检索（各取 top-20）    │
│                                      │
│  Chroma 语义检索    BM25 关键词检索   │
│  (bge-m3 1024-dim)  (jieba 分词)     │
│        │                  │          │
│        └──────┬───────────┘          │
│               ▼                      │
│       Union 去重 + 归一化排序          │
│         ~25-35 篇候选                 │
└─────────────────────────────────────┘
    │ 候选文档列表
    ▼
┌─────────────────────────────────────┐
│  Step 3: Reranker 精排               │
│  gte-rerank-v2 API (Cross-Encoder)   │
│  逐对打分 → 按相关性降序 → top-3      │
└─────────────────────────────────────┘
    │ 3 篇最相关文档
    ▼
┌─────────────────────────────────────┐
│  LLM 生成回答                        │
│  文档注入 prompt 作为参考资料          │
└─────────────────────────────────────┘
```

---

## 二、各组件说明

| 组件 | 技术方案 | 作用 | 关键参数 |
|------|---------|------|---------|
| **Query Rewriting** | 自定义 Prompt + qwen3.7-max | 口语→术语，跨术语鸿沟 | 10 条示例覆盖全部 9 类，规则 4 防误映射 |
| **Chroma 语义检索** | bge-m3 via ollama 本地 | 语义相似度召回 | 1024 维，top-20 |
| **BM25 关键词检索** | rank-bm25 + jieba 分词 | 关键词精确匹配 | top-20 |
| **Union 合并** | 分数归一化 + ID 去重 + 降序排列 | 合并双路结果，防止单路霸占前排 | — |
| **Reranker 精排** | gte-rerank-v2 API（百炼平台） | Cross-Encoder 逐对打分，精选 top-3 | top_n=20, top_k=3 |

---

## 三、关键设计决策

### 3.1 为什么不用 RRF 融合，改用 Union + Reranker？

| 方案 | 141 条合成数据 Recall@3 | 15 条真实数据 Recall@3 |
|------|------------------------|----------------------|
| RRF 融合 (ADR 0011 旧方案) | 0.567 | 0.133 |
| Union + Reranker (v3.3) | **0.723** | **0.222** |

RRF 给双路等权投票。当 Chroma 返回噪声时（通用 embedding 跨不过术语鸿沟），噪声和 BM25 的好结果"对等表决"，排序被污染。Reranker 是 Cross-Encoder，独立评估每个 query-doc 对，不受候选池里其他文档影响。

### 3.2 为什么保留 Chroma 但不主导？

| 检索方式 | 141 条合成 | 15 条真实 |
|---------|-----------|----------|
| BM25 only | 0.312 | 0.178 |
| Chroma only | 0.553 | 0.200 |

合成数据上 Chroma 远超 BM25（术语天然匹配），真实数据上两者持平。Chroma 对真实查询暂不贡献增益但不拖后腿——Reranker 会自动过滤低质候选。当未来有心理学领域微调 embedding 时，Chroma 可无缝启用。

### 3.3 为什么 QR 只在 knowledge 路由触发？

Router 判断用户意图后分四种路由：

| 路由 | 场景 | 是否走 RAG | 是否走 QR |
|------|------|-----------|----------|
| crisis | 危机干预 | ❌ | ❌ |
| comfort | 情感安抚 | ❌ | ❌ |
| general | 闲聊 | ❌ | ❌ |
| knowledge | 知识咨询 | ✅ | ✅ |

QR 是 LLM 调用，按需触发，闲聊不会浪费 token。

### 3.4 为什么 top_n=20, top_k=3？

- **top_n=20**：BM25 和 Chroma 各多捞几条，扩大候选池覆盖。从 15 提到 20，成本增幅极小（API 按 token 计费），但可能捞出排名 15-20 的好文档。
- **top_k=3**：最终注入 LLM prompt 的文档数。3 篇在信息量和 prompt 长度之间取得平衡；实测 K=5 多出来的文档质量不差，但 token 消耗增加 60%+。

---

## 四、Query Rewriting Prompt 设计

### 4.1 十条示例覆盖九个知识类别

| 示例 | 覆盖类别 |
|------|---------|
| 男朋友躲着我不回消息 | relationships |
| 什么都往坏处想 | psychology_basics / clinical |
| 睡不着脑子里停不下来 | sleep_health |
| 做什么都没意思提不起劲 | disorder_knowledge (depression) |
| 分手后走不出来 | grief_and_loss |
| 人多就紧张出汗 | disorder_knowledge (anxiety) |
| 担心自己得重病 | medication_knowledge / clinical |
| 迟到就觉得自己一无是处 | psychology_basics |
| 经历过不好的事，心慌手抖 | **trauma_and_stress** ← v3.3 新增 |
| 不知道怎么调节自己 | **coping_strategies** ← v3.3 新增 |

### 4.2 四条核心规则

1. 保留原意，补充心理学专业术语和同义词
2. 不编造用户没说的症状或情绪
3. 只输出关键词，不要完整句子，不要解释
4. 将具体场景词（如"迟到"、"打翻水杯"）映射为背后的心理过程词（如"失败情境"、"失误"），但绝不改写为"时间管理"或"拖延"

---

## 五、配置项

### 5.1 环境变量

| 变量 | 值 | 用途 |
|------|---|------|
| `REWRITER_MODEL_NAME` | `qwen3.7-max` | 查询改写 LLM |
| `EMBEDDING_MODEL_NAME` | `bge-m3` | 向量嵌入模型 |
| `EMBEDDING_API_BASE` | `http://localhost:11434/v1` | ollama 地址 |
| `EMBEDDING_BACKEND` | `api` | OpenAI 兼容 API 透传 ollama |
| `RERANK_MODEL_NAME` | `gte-rerank-v2` | 百炼重排序模型 |
| `RRF_DENSE_WEIGHT` | `1.0` | Chroma 稠密路权重（Union 归一化后影响较小） |

### 5.2 代码常量

| 位置 | 参数 | 值 |
|------|------|---|
| `service.py` | Reranker top_n | 20 |
| `service.py` | Reranker top_k | 3 |
| `hybrid_retriever.py` | RRF_K | 60（仅 RRF 兜底时使用） |

---

## 六、代码文件清单

```
core/rag/
├── query_rewriter.py       # LLM 查询改写（10 条心理领域示例 + 4 条规则）
├── reranker.py             # gte-rerank-v2 API（+ BGE 本地备用 + qwen3-rerank 可切换）
├── hybrid_retriever.py     # 双路检索 + Union 合并 + RRF 兜底
├── chroma_store.py         # Chroma 向量库封装
├── bm25_index.py           # BM25 关键词索引
├── embedder.py             # Embedding 工厂（bge-m3 / API 可切换）
├── external_retriever.py   # 外部搜索 fallback
└── query_classifier.py     # 已废弃 (ADR 0011)

modules/intervention/rag/
└── retriever.py            # KnowledgeRetriever：统一检索入口

modules/intervention/
└── service.py              # 注入 QR + Reranker，按 route 分发

scripts/
├── test_rag_full.py        # RAG 全链路交互测试
├── eval_query_rewriter.py  # 改写器量化评测
├── eval_retrieval.py       # 全量检索评测
└── build_knowledge_index.py # Chroma 索引构建
```

---

## 七、评测命令

```powershell
conda activate emotion
cd e:\postgrad\PRISM-master

# 交互式测试
python scripts/test_rag_full.py "你的问题" -k 3 5

# 全量评测
python scripts/eval_retrieval.py
```

---

## 八、已知局限与未来方向

1. **术语鸿沟**：用户口语 vs 临床术语是当前最大瓶颈。Query Rewriting 弥补约 50%，仍有 70% 相关文档无法命中。方向：领域微调 embedding 模型。
2. **Chroma 贡献为零**：当前通用 embedding 对真实查询无增益。方向：换心理学专用 embedding 或给文档补口语化字段。
3. **知识库覆盖度**：860 篇文档对 trauma_and_stress、coping_strategies 覆盖不足。方向：持续扩充知识库。
4. **QR 缓存**：同一查询重复调用 LLM 改写。方向：按原文 hash 缓存改写结果。
