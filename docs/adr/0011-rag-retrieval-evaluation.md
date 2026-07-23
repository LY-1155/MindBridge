# RAG 检索层评估体系：合成数据 + 指标体系 + 消融实验

> **⚠️ v3.3 更新（2026-07-17）**：本文评估的 RRF 融合方案（hybrid_bare=0.567）已被 **Union + Reranker** 超越（0.723，+27%）。
> 评测方法（Recall@3/MRR/HitRate@3）和测试集仍在用，但 pipeline 架构已升级。
> 详见 [RAG-v3.3-设计文档](../RAG-v3.3-设计文档.md) 和 [RAG优化记录](../RAG优化记录.md)。

建立 RAG 检索层的量化评估体系，通过合成测试数据生成和标准化指标（Recall@3, MRR, Hit Rate@3）衡量检索质量，配合消融实验分解各组件贡献。同时修复 QueryClassifier 的 category 分类结果未被检索层使用的问题。

## 为什么

### 为什么不直接评估端到端生成质量

- RAG 管线分三层：Query 分类 → 检索 → 生成。生成是检索的下游——检索召回了不相关文档，Gen 不可能好
- 检索层有明确的 ground truth（哪些文档该被召回），评估标准成熟、争议少
- 生成层评估需要 LLM-as-judge 评判 faithfulness，引入评判 LLM 的偏差，解释性差
- 分层评估意味着"检索变差但生成变好"这类混淆信号可以被拆解

### 为什么用合成数据（B 方案）而非人工标注

- 人工标注需要标注者逐篇阅读 ~860 篇心理健康知识文档后写查询，时间成本过高
- 合成方案：LLM 从每篇文档反向生成"这条文档能回答的用户问题"，document → query 的映射天然形成 ground truth
- 心理健康领域用户咨询高度口语化、自然化，LLM 有足够的能力生成接近真实的咨询句式
- 实测中同样会用手写查询做 LLM-as-judge 补充验证（C 方案），弥补合成数据措辞偏正式的缺陷

### 为什么自定义脚本而非 ragas/deepeval

- 本项目检索架构有自有的层次（bare retriever vs full pipeline）、自定义消融维度（Chroma-only / BM25-only / Hybrid），现成框架需大量适配
- 所需指标仅三个（Recall@3, MRR, Hit Rate@3），自己实现不超过 30 行
- 评测脚本需直接访问内部组件（HybridRetriever, BM25Index, ChromaStore）做分组对照，框架抽象层反成障碍
- 零额外依赖，CI 可直接通过 `--dry-run` 用假组件运行

### 为什么选 Recall@3 + MRR + Hit Rate@3 而不是 Precision@3 / NDCG

- **Recall@3** 是天花板指标——系统只取 top-3 注入 LLM prompt，没召回的东西 Gen 永远看不到。这是最重要的指标
- **MRR** 回答"第一个好答案在哪"——top-3 里第一个结果决定了用户（和 LLM）的第一印象
- **Hit Rate@3** 是宽容指标——只要有 1 条相关文档命中，管道就能产生有效输出
- **Precision@3** 对合成数据意义有限：每条查询只对应 1-3 篇"该中"的文档，top-3 里全中实际只意味着"覆盖了那 1 篇"，并不代表检索质量优秀
- **NDCG@k** 需要相关性分档（0/1/2/3），合成数据的 binary ground truth 无法支撑；项目也不做重排序，排序质量差异非关注重点

### 为什么做消融实验（Chroma-only / BM25-only / Hybrid）

- Hybrid = Chroma dense + BM25 sparse → RRF 融合。如果整体指标差，无法判断是哪个组件的问题
- 心理健康领域有明显的 query 特征分化：药物名查询偏关键词匹配（BM25 强），"心里堵得慌"等情感表达偏语义检索（Chroma 强）
- 消融结果可直接指导后续优化方向——是换 embedding 模型还是调 BM25 参数

### 为什么拆成裸检索器 vs 完整管线两组

- 裸检索器（无 QueryClassifier）测纯 Chroma/BM25/RRF 能力，对标学术界标准
- 完整管线测分类器对检索的增益/衰减——分类错误会导致相关文档被 source/category 过滤挡在门外
- 两组指标差异直接量化分类器的噪声水平，避免检索和分类混淆评估

### 为什么要修复 category 过滤

- `QueryClassifier` 调用 LLM 对用户查询做 source + categories 分类（9 类），但 `_build_filter` 中原实现将 categories 丢弃，只使用 source
- 这意味着 LLM 分类器消耗 token 的输出约 50% 被浪费
- 修复后 categories 过滤会同时作用在 Chroma（$in 操作符）和 BM25（逐 category 查询合并）两路
- 修复后完整管线的指标才有实际意义——否则完整管线与裸检索器的差异仅体现在 source 过滤上

## 架构

### 评估组件

```
scripts/
├── generate_rag_test_data.py    ← B 方案：从 JSONL 文档反向生成查询
└── eval_retrieval.py            ← 主评估脚本：6 组对照 + 3 指标 + 消融

data/eval/
├── synthetic_queries.jsonl      ← 生成产物（纳入 .gitignore）
└── judge_queries.jsonl          ← 手写查询（纳入 git）

core/rag/
├── hybrid_retriever.py          ← 新增 retrieve_with_ids() 和消融方法
└── query_classifier.py          ← 无变更
```

### 合成数据生成流程

```
知识库 JSONL 文档 (~860 篇)
    │
    │ 抽样 N 篇（默认 50）
    │ 每批 5 篇发给 LLM
    ▼
LLM 反向生成查询
    │ "舍曲林停药后为什么会头晕"
    │ "SSRI 药物的常见副作用有哪些"
    ▼
data/eval/synthetic_queries.jsonl
    {"query": "...", "relevant_doc_ids": ["doc_x"], "expected_categories": [...], "expected_source": "public"}
```

### 6 组对照实验矩阵

| # | 组名 | 检索方式 | QueryClassifier | RRF 融合 | 目的 |
|---|---|---|---|---|---|
| 1 | dense_bare | Chroma-only, 无过滤 | 否 | 否 | 纯稠密向量基线 |
| 2 | sparse_bare | BM25-only, 无过滤 | 否 | 否 | 纯关键词基线 |
| 3 | dense_full | Chroma + classifier 过滤 | 是 | 否 | 分类器对稠密检索的影响 |
| 4 | sparse_full | BM25 + classifier 过滤 | 是 | 否 | 分类器对关键词检索的影响 |
| 5 | hybrid_bare | 两路 + RRF（无分类器） | 否 | 是 | RRF 融合自身的基线 |
| 6 | hybrid_full | 两路 + RRF + 分类器 | 是 | 是 | 完整管线（最终系统） |

组间对比可以回答：
- full vs bare 的 Delta = 分类器过滤的价值
- hybrid vs dense/sparse 的 Delta = RRF 融合的增益
- dense vs sparse = 语义 vs 关键词各领域的强弱

### 指标定义

```python
Recall@3  = |top-3_ids ∩ relevant_ids| / |relevant_ids|
MRR       = 1 / rank_of_first_relevant  （0 if no hit）
Hit Rate@3 = 1 if any relevant in top-3, else 0  （averaged across queries）
```

### HybridRetriever 新增方法

```python
def retrieve_with_ids(self, query, top_k) -> List[dict]
    # 返回 [{"id": "doc_xxx", "text": "...", "score": 0.85}, ...]
    # 评估用，需要 doc ID 做 ground truth 匹配

def retrieve_dense_only(self, query, top_k) -> List[dict]
    # Chroma-only，无分类器，无 RRF。消融组 1

def retrieve_sparse_only(self, query, top_k) -> List[dict]
    # BM25-only，无分类器，无 RRF。消融组 2
```

### category 过滤修复

`_build_filter` 将 `classification.categories` 传入 filter dict：

- **Chroma 路**：通过 ChromaDB `$in` 操作符 `{"category": {"$in": ["coping_strategies", "sleep_health"]}}` 传给 `where` 子句
- **BM25 路**：逐 category 单独调用 `BM25Index.search(filter_meta={"category": cat})`，合并去重保留最高分

## 备选方案

### ragas / deepeval 框架

- ragas 的检索层指标（Context Relevance, Context Recall）依赖 LLM-as-judge，不提供传统的 Recall@k/MRR
- deepeval 需要按框架接口重新包装检索器，6 组消融对照会变成 6 次框架实例化
- 两个框架的核心价值在生成层评估（Faithfulness, Answer Relevancy），检索层反而增加复杂度
- 决策：暂不使用，后续如需扩展至生成层评估再引入

### 人工标注 ground truth

- 需要标注入花时间通读全部 ~860 篇文档，标记每个 query 的 relevant doc 集合
- 质量最高，但对当前项目阶段投入产出比低
- 决策：暂不采用，合成数据 + LLM-as-judge 组合覆盖检索层量化需求

## 影响

- `HybridRetriever` 新增 3 个公开方法，`retrieve()` 内部重构调用 `retrieve_with_ids()`，对外接口不变
- 新增 `scripts/generate_rag_test_data.py` 和 `scripts/eval_retrieval.py` 两个独立脚本，不影响现有管线
- `QueryClassifier.categories` 的分类结果从被丢弃变为实际参与过滤，对检索结果有实质改变——需跑评估确认分类器质量
- 如分类器 categories 错误率高，评估脚本会通过 dense_full vs dense_bare 对比暴露出来

## 评估结果

### Round 1: nomic-embed-text (768-dim) — 2026-07-08, 20 queries

nomic-embed-text 为英文模型，对中文知识库产生的向量基本随机。

| Group | Recall@3 | MRR | HitRate@3 |
|---|---|---|---|
| dense_bare | 0.05 | 0.05 | 0.05 |
| sparse_bare | 0.40 | 0.33 | 0.40 |
| dense_full | 0.05 | 0.05 | 0.05 |
| sparse_full | 0.30 | 0.25 | 0.30 |
| hybrid_bare | 0.25 | 0.15 | 0.25 |
| hybrid_full | 0.15 | 0.13 | 0.15 |

结论：Dense 基本失效 (0.05 ≈ 随机)，BM25 是唯一可用检索方式。切 bge-m3。

### Round 2: bge-m3 (1024-dim) — 2026-07-08, 20 queries

| Group | Recall@3 | MRR | HitRate@3 |
|---|---|---|---|
| dense_bare | **0.65** | 0.54 | 0.65 |
| sparse_bare | 0.40 | 0.33 | 0.40 |
| dense_full | 0.40 | 0.38 | 0.40 |
| sparse_full | 0.30 | 0.25 | 0.30 |
| hybrid_bare | **0.70** | 0.59 | 0.70 |
| hybrid_full | 0.30 | 0.30 | 0.30 |
| full-bare Delta | -0.40 | -0.29 | -0.40 |

消融分析：
- Hybrid vs Dense-only (bare): Recall +0.05 — RRF 融合有轻微增益
- Hybrid vs Sparse-only (bare): Recall +0.30 — 稠密语义检索弥补了关键词盲区
- Classifier gain on Dense: -0.25 — 分类器误杀正确结果
- Classifier gain on Sparse: -0.10 — 分类器对关键词检索影响较小

结论：
1. bge-m3 让 dense 从 0.05 → 0.65，验证了 nomic-embed-text 不适用于中文
2. hybrid_bare (RRF without classifier) = 0.70 是最佳方案
3. QueryClassifier 的 category 过滤对 Recall 有显著危害 (-0.40)，建议生产环境暂不启用 category 过滤，或提升分类器准确率后再启用

### Round 3: bge-m3 (1024-dim) 全量 — 2026-07-08, 141 queries

| Group | Recall@3 | MRR | HitRate@3 |
|---|---|---|---|
| dense_bare | **0.553** | 0.422 | 0.553 |
| sparse_bare | 0.312 | 0.262 | 0.312 |
| dense_full | 0.383 | 0.330 | 0.383 |
| sparse_full | 0.262 | 0.222 | 0.262 |
| hybrid_bare | **0.567** | 0.467 | 0.567 |
| hybrid_full | 0.369 | 0.312 | 0.369 |
| full-bare Delta | -0.199 | -0.155 | -0.199 |

消融分析：
- Hybrid vs Dense-only (bare): Recall +0.014 — RRF 融合增益极小
- Hybrid vs Sparse-only (bare): Recall +0.255 — dense 对 sparse 有显著补偿
- Classifier gain on Dense: -0.170 — 分类器损害
- Classifier gain on Sparse: -0.050 — 对关键词检索影响较小

按知识类别分组 (Recall@3):

| Category | dense_bare | sparse_bare | hybrid_full |
|---|---|---|---|
| clinical | 0.583 | 0.188 | 0.271 |
| coping_strategies | 0.833 | 0.833 | 0.833 |
| disorder_knowledge | 0.833 | 0.500 | 0.000 |
| grief_and_loss | 0.667 | 1.000 | 1.000 |
| medication_knowledge | 1.000 | 0.833 | 0.000 |
| psychology_basics | 0.111 | 0.333 | 0.667 |
| relationships | 0.083 | 0.417 | 0.750 |
| trauma_and_stress | 0.667 | 0.667 | 1.000 |

结论：
1. hybrid_bare = 0.567 为最佳方案，dense 单一检索（0.553）已接近，BM25 RRF 融合增益仅 +0.014
2. 分类器效果高度分化：对 medication/disorder 完全误杀（1.0→0.0），对 relationships/psychology_basics 有显著提升（0.08→0.75 / 0.11→0.67）
3. 全量数据比 20-query 样本低约 0.13，20 条采样存在正向偏差
4. 建议：短期上 hybrid_bare（关分类器），长期针对 medication/disorder 类修复分类器规则后再启用

### 决策：移除生产管线中的 QueryClassifier — 2026-07-08

基于 Round 3 全量评估结果，决定从生产检索管线中移除 QueryClassifier 的 category 过滤：

- `HybridRetriever.__init__` 的 `classifier` 参数改为可选（`Optional[QueryClassifier] = None`）
- `retrieve_with_ids()` 不再调用分类器，直接 dense + BM25 → RRF 融合
- 生产 `KnowledgeRetriever` 移除了 `QueryClassifier` 依赖和 `llm` 参数
- 评估脚本保留分类器，用于消融实验中的 full 组对照

理由：
1. hybrid_full（含分类器）Recall@3=0.369，hybrid_bare（无分类器）Recall@3=0.567，分类器造成 -0.199 的损失
2. 860 篇文档规模下，bge-m3 语义检索的区分度已足够，不需要预过滤缩小范围
3. 分类器消耗额外 LLM 调用且增加延迟，投入产出比为负
4. 分类器代码保留，未来知识库扩至数万篇时可按需重新启用
