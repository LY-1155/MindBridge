# 知识库检索从逐层回退改为统一索引 + 混合检索

> **⚠️ v3.3 更新（2026-07-17）**：本文描述的 RRF 融合方案已被 **Union + Reranker** 替代。
> RRF 在双路质量不均衡时 1+1<1，详见 [RAG-v3.3-设计文档](../RAG-v3.3-设计文档.md)。
> 本文保留作为统一索引和双路检索的架构决策记录，融合方式部分已过时。

知识路由下的知识检索原设计为三层逐级回退（医院私有 → 公有心理学 → 外部 API），现改为私有+公有统一 Chroma 索引、稠密向量 + BM25 双路混合检索、LLM 分类做 metadata 过滤。

## 为什么

- **逐层回退有延迟浪费**。私有 500 条覆盖临床对话场景，但用户问"SSRI 副作用"时私有库不可能有答案，第一层必然空搜，白白消耗一次 embedding API 调用和网络往返。
- **私有和公有的知识域互补**。私有覆盖临床技法、家庭治疗经验；公有覆盖药物、量表解读、诊断标准等。用户的问题经常需要跨域检索，不应该人为制造 barrier。
- **云端 embedding 没有稀疏向量**。百炼 API 只返回稠密向量，缺少 bge-m3 自带的 sparse 输出，需要本地 BM25 补上精确关键词匹配这条腿。
- **LLM 分类比规则词典更准确**。"我什么都提不起兴趣"是 disorder 方向，规则匹配不到"抑郁"关键词就会误分类。

## 考虑过的方案

- **保持逐层回退**：不改架构，只在每层内升级检索质量。优点是不动现有逻辑，缺点是私有层空搜的延迟无法消除，且跨域查询体验差。
- **统一索引不加 BM25**：纯稠密检索。优点是简单，缺点是精确术语匹配（药名、量表名）没有保障，稠密向量的语义近似可能把"氟西汀"和"帕罗西汀"混到一起。
- **LLM 分类改为规则分类**：零延迟但覆盖率低。用户口语化表达（"心里堵得慌""什么都提不起劲"）不可能用穷举规则覆盖。不选。
- **本地 bge-m3 替代百炼**：可获得 sparse 向量，省掉 BM25。但考虑项目需部署到服务器，百炼 API 统一管理凭证和计费，运维成本更低。不选。

## 影响

- `modules/intervention/rag/retriever.py` 的 `KnowledgeRetriever` 需重写，从逐层 `_retrieve_local` 改为统一的 `HybridRetriever`
- 新增 `core/rag/` 目录：`embedder.py`、`bm25_index.py`、`hybrid_retriever.py`
- 新增 `scripts/build_knowledge_index.py`：一键将 JSONL 数据建索引到 Chroma
- `data/knowledge/sources.json` 的 layers 配置语义从逐层回退变为元数据分类声明
- 新增 `schemas/contracts/`：`QueryClassification` 定义 LLM 分类输出的 category 和 source 字段
