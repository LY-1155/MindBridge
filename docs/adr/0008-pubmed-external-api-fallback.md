# 外部 API 兜底从占位改为 PubMed E-utilities 文献检索

> **⚠️ v3.3 更新（2026-07-17）**：本文中 fallback 触发条件"RRF top-1 score < 0.5"已过时。
> v3.3 不再使用 RRF 融合，外部 API fallback 的触发逻辑需重新评估。
> 详见 [RAG-v3.3-设计文档](../RAG-v3.3-设计文档.md)。

external_api 层原为空白占位（未接入），现接入 NCBI PubMed E-utilities 作为本地知识库分数不足时的兜底检索源。

## 为什么

- **老师要求接入外部 API**。当前 external_api layer enabled=false、base_url=""，没有任何代码实现。
- **PubMed 覆盖本地知识库的弱项**。药物副作用、剂量、药物相互作用、罕见诊断——这些是 8 类公有知识库中信息密度最高的子集，但初建阶段数据可能不完整。PubMed 有数百万篇生物医学文献摘要，是天然的安全网。
- **免费 + 无需强制 API Key**。E-utilities 不带 key 限 3 次/秒，注册 key 后 10 次/秒。对于非高频查询的系统足够。
- **不做每一次都检索**。PubMed 检索延迟 300~500ms，每次都跑是在浪费。只在本地 RRF 融合后 top-1 score < 0.5 时触发，作为兜底。

## 触发机制

```
用户查询
  │
  ├─ QueryClassifier 分类
  ├─ Chroma 稠密检索 (category metadata 过滤)
  ├─ BM25 关键词检索 (category metadata 过滤)
  │
  ├─ RRF 融合
  │     │
  │     ├─ top-1 score ≥ 0.5 → 不触发 PubMed，直接返回
  │     └─ top-1 score < 0.5 → 触发 PubMed 兜底
  │
  └─ PubMed 结果追加到 RRF 结果集，最终取 top-3
```

## PubMed 接口

两步都是 GET 请求，无需认证（不带 key 3次/秒，带 key 10次/秒）：

### Step 1: esearch — 按关键词搜 PMID

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
    ?db=pubmed
    &retmax=5
    &sort=relevance
    &term=<keywords>
```

返回 `<IdList>` 中 5 个 PMID。

### Step 2: efetch — 拿摘要文本

```
GET https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
    ?db=pubmed
    &id=<pmid1,pmid2,...>
    &rettype=abstract
    &retmode=xml
```

提取 `<ArticleTitle>` + `<AbstractText>` 拼接为文本片段。

## 查询预处理（中文 → 关键词）

PubMed 对中文自然语言查询效果差（"吃了舍曲林之后恶心想吐怎么办"大概率空结果），需要预处理。不调 LLM 翻译（费 token 费延迟），复用系统已有的 jieba 分词：

```
输入: "我吃了舍曲林之后总是恶心想吐怎么办"
jieba 分词 → 我 / 吃了 / 舍曲林 / 之后 / 总是 / 恶心 / 想吐 / 怎么办
保留名词+英文词 → 舍曲林 恶心
关键词串发给 PubMed esearch term 参数
```

英文词（如"SSRI""CBT""PHQ-9"）在 jieba 词性中可能被标为 eng，一并保留。

## 降级策略

三层保护，PubMed 故障不影响主链路：

1. **超时**（5 秒）→ 静默丢弃，只返回本地结果
2. **返回空结果**（无 PMID 或无摘要）→ 静默丢弃
3. **网络错误**（DNS/连接失败）→ 静默丢弃，记录 warning 日志

PubMed 结果作为低权重文本片段追加到检索结果中（无 RRF rank，固定追加在末尾），不抢占本地高质量条目的位置。

## 考虑过的方案

- **每次检索都跑 PubMed（RRF 第三路）**：延迟增加 300~500ms/次，且 PubMed 对哀伤、人际关系等场景搜不出有用结果，浪费。不选。
- **LLM 翻译查询为英文再搜**：翻译 + 检索串行延迟更长，且 PubMed 其实收录大量中文文献（中文标题+英文摘要），中文关键词就可以命中。不选。
- **按 category 白名单触发**：只有 medication_knowledge、disorder_knowledge 才跑 PubMed。优点是精细，但初始阶段 8 类数据都不完整，用分数阈值更通用。暂不选，未来数据充实后可考虑切换为白名单模式。
- **DuckDuckGo / Wikipedia 替代**：Wikipedia 中文心理学条目质量参差不齐，DuckDuckGo 返回的是网页摘要而非经过同行评审的文献。PubMed 的文献摘要质量更高且可溯源。不选。

## 与 Tavily 药物搜索 fallback 的关系 (ADR 0009)

本 ADR 设计的 PubMed 兜底是**分数阈值驱动**的通用方案（RRF top-1 score < 0.5 时触发，面向全部 8 个公有类别），目前**尚未实现**。

ADR 0009 中已实现的 Tavily 药物搜索 fallback 是**药物名感知驱动**的定向方案：当 `HybridRetriever` 通过 `ExternalRetriever` 检测到查询中提到了知识库未收录的精神科药物时，通过 Tavily Search API 实时检索 `site:dayi.org.cn`，补充药物信息。覆盖 50+ 种精神科药物。

二者共用 `core/rag/external_retriever.py` 作为集成入口，在同一层（`HybridRetriever.retrieve()` 中 RRF 融合后）调用：

| 维度 | PubMed (未实现) | Tavily (已实现) |
|------|----------------|-----------------|
| 触发条件 | RRF top-1 score < 0.5 | 查询中含知识库未收录的精神科药物 |
| 覆盖范围 | 全部 8 个公有类别 | 精神科药物 |
| 信源 | PubMed 文献摘要（同行评审） | dayi.org.cn 药品说明书 |
| 延迟 | 300~500ms | 500~1000ms（含 0.5s 节流） |
| 集成点 | ExternalRetriever → HybridRetriever.retrieve() | ExternalRetriever → HybridRetriever.retrieve() |

二者在同一个集成点按序执行：Tavily 先跑（药物名感知，定向补充），PubMed 后跑（分数阈值，通用兜底），结果合并后去重。

文件分工：
- `core/rag/search_fallback.py`：`DrugNameMatcher` + `TavilySearchProvider`（底层实现）
- `core/rag/external_retriever.py`：`ExternalRetriever`（配置驱动的统一入口，当前仅 Tavily 一个 provider）
- `core/rag/hybrid_retriever.py`：在 `retrieve()` 中 RRF 融合后调用 `ExternalRetriever.search()`

## 影响

- 已有 `core/rag/external_retriever.py`：`ExternalRetriever` 类，从 sources.json 加载配置并封装 fallback 逻辑（当前仅 Tavily provider）
- 已有 `core/rag/search_fallback.py`：`DrugNameMatcher` + `TavilySearchProvider` 底层实现
- 已有 `core/rag/hybrid_retriever.py`：`retrieve()` 中 RRF 融合后调用 `ExternalRetriever.search()`
- 已有 `data/knowledge/sources.json`：v4，external_api 层启用 Tavily
- 新增 `core/rag/external_retriever.py`：`_search_pubmed` 方法作为 provider 的一种，在 `ExternalRetriever` 中注册
- 修改 `data/knowledge/sources.json`：external_api 层支持多 provider（tavily + pubmed）
- 可选修改 `.env`：新增 `PUBMED_API_KEY`、`PUBMED_TOOL_NAME`（E-utilities 规范要求 User-Agent 包含工具名）
- 检索延迟：PubMed 未触发时无任何影响；触发时增加 300~500ms（含超时保护）
