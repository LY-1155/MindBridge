"""混合检索器：稠密向量 + BM25 → RRF 融合 → 外部 API fallback → top-k 文本片段"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from core.rag.bm25_index import BM25Index
from core.rag.chroma_store import ChromaStore
from core.rag.query_classifier import QueryClassifier, QueryClassification

if TYPE_CHECKING:
    from core.rag.external_retriever import ExternalRetriever

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 融合参数


class HybridRetriever:
    """混合检索器

    流程：LLM 分类 → Chroma 稠密 + BM25 关键词 → RRF 融合 → 外部 API fallback → top-k 文本
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        bm25_index: BM25Index,
        classifier: Optional[QueryClassifier] = None,
        source_weights: Optional[Dict[str, float]] = None,
        external_retriever: Optional["ExternalRetriever"] = None,
        dense_weight: float = 1.0,
    ):
        self._chroma = chroma_store
        self._bm25 = bm25_index
        self._classifier = classifier
        self._source_weights = source_weights or {}
        self._external = external_retriever
        self._dense_weight = dense_weight

    # ── 公开 API ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """主检索入口，返回文本片段列表"""
        combined = self.retrieve_with_ids(query, top_k=top_k)
        results = [item["text"] for item in combined]

        # 外部 API fallback（知识库未覆盖内容 → Tavily 实时搜索）
        if self._external is not None:
            try:
                supplements = self._external.search(query, results)
                if supplements:
                    results = results + supplements
                    logger.info("外部 API fallback 命中: query=%.50s..., 补充 %d 条",
                                query, len(supplements))
            except Exception:
                logger.warning("外部 API fallback 执行失败", exc_info=True)

        return results

    def retrieve_with_ids(self, query: str, top_k: int = 3) -> List[dict]:
        """检索入口，返回带 doc ID 的结构化结果。用于评估。

        Returns:
            [{"id": "doc_xxx", "text": "...", "score": 0.85}, ...]
        """
        if not query.strip():
            return []

        # 双路检索（无分类器过滤）
        dense_docs = self._search_chroma(
            query, top_k=top_k * 3, chroma_filter={})
        bm25_docs = self._search_bm25(
            query, top_k=top_k * 3, bm25_filter={})

        # RRF 融合
        return self._rrf_fuse(dense_docs, bm25_docs, top_k=top_k)

    def retrieve_dense_only(self, query: str, top_k: int = 3) -> List[dict]:
        """Chroma-only 检索，无分类器过滤、无融合。消融实验用。

        Returns:
            [{"id": "doc_xxx", "text": "...", "score": 0.85}, ...]
        """
        if not query.strip():
            return []
        return self._search_chroma(query, top_k=top_k, chroma_filter={})

    def retrieve_sparse_only(self, query: str, top_k: int = 3) -> List[dict]:
        """BM25-only 检索，无分类器过滤、无融合。消融实验用。

        Returns:
            [{"id": "doc_xxx", "text": "...", "score": 0.85}, ...]
        """
        if not query.strip():
            return []
        return self._search_bm25(query, top_k=top_k, bm25_filter={})

    def retrieve_union_with_ids(self, query: str, top_k: int = 15) -> List[dict]:
        """双路并集检索：Chroma + BM25 → 去重合并，不做 RRF 融合。

        用于配合 Reranker 的检索流程：两路各取 top_k 条，合并去重后
        交由 Reranker 精排。采用分数归一化后降序排列，避免单路霸占前排。

        Returns:
            [{"id": "...", "text": "...", "score": ...}, ...]
        """
        if not query.strip():
            return []

        dense_docs = self._search_chroma(query, top_k=top_k, chroma_filter={})
        bm25_docs = self._search_bm25(query, top_k=top_k, bm25_filter={})

        # 各路分数归一化到 [0,1]
        def _normalize(docs: list) -> list:
            if not docs:
                return docs
            scores = [d["score"] for d in docs]
            s_min, s_max = min(scores), max(scores)
            if s_max == s_min:
                return docs
            for d in docs:
                d["norm_score"] = (d["score"] - s_min) / (s_max - s_min)
            return docs

        dense_docs = _normalize(dense_docs)
        bm25_docs = _normalize(bm25_docs)

        # 合并去重，按归一化分数降序
        seen: set = set()
        results: List[dict] = []
        for doc in sorted(dense_docs + bm25_docs,
                          key=lambda d: d.get("norm_score", 0),
                          reverse=True):
            if doc["id"] not in seen:
                seen.add(doc["id"])
                results.append(doc)

        return results

    # ── 内部分类与过滤 ──────────────────────────────────────

    def _build_filter(self, classification: QueryClassification) -> dict:
        """根据分类结果构建过滤条件"""
        filt: dict = {}
        src = classification.source

        if src == "private":
            filt["source"] = "private"
        elif src == "public":
            filt["source"] = "public"
        # src="both" 或 "all" → 不过滤 source

        if classification.categories:
            filt["categories"] = list(classification.categories)

        return filt

    # ── 双路检索实现 ────────────────────────────────────────

    def _search_chroma(
        self,
        query: str,
        top_k: int,
        chroma_filter: dict,
    ) -> List[dict]:
        """Chroma 稠密检索，支持 source + category 过滤"""
        conditions = []
        if "source" in chroma_filter:
            conditions.append({"source": chroma_filter["source"]})
        if "categories" in chroma_filter and chroma_filter["categories"]:
            conditions.append({"category": {"$in": chroma_filter["categories"]}})

        # ChromaDB where 要求单操作符；多个条件用 $and 组合
        if len(conditions) == 0:
            chroma_where = None
        elif len(conditions) == 1:
            chroma_where = conditions[0]
        else:
            chroma_where = {"$and": conditions}

        results = self._chroma.search(
            query=query,
            top_k=top_k,
            filter_meta=chroma_where,
            source_weights=self._source_weights,
        )
        return results

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        bm25_filter: dict,
    ) -> List[dict]:
        """BM25 关键词检索，支持 source + 多 category 合并"""
        categories = bm25_filter.get("categories", [])
        source = bm25_filter.get("source")

        if not categories:
            # 无 category 过滤：只用 source 查一次（原有行为）
            src_filter = {"source": source} if source else None
            raw = self._bm25.search(query, top_k=top_k, filter_meta=src_filter)
            return [
                {"id": doc_id, "text": self._bm25._docs.get(doc_id, ""), "score": score}  # noqa: SLF001
                for doc_id, score in raw
            ]

        # 多 category：每个 category 单独查，合并去重（保留最高分）
        all_results: Dict[str, Tuple[str, float]] = {}
        for cat in categories:
            cat_filter = {"category": cat}
            if source:
                cat_filter["source"] = source
            raw = self._bm25.search(query, top_k=top_k, filter_meta=cat_filter)
            for doc_id, score in raw:
                if doc_id not in all_results or score > all_results[doc_id][1]:
                    text = self._bm25._docs.get(doc_id, "")  # noqa: SLF001
                    all_results[doc_id] = (text, score)

        sorted_results = sorted(all_results.items(), key=lambda x: x[1][1], reverse=True)
        return [
            {"id": doc_id, "text": text, "score": score}
            for doc_id, (text, score) in sorted_results[:top_k]
        ]

    # ── 融合 ──────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense_docs: List[dict],
        bm25_docs: List[dict],
        top_k: int = 3,
    ) -> List[dict]:
        """RRF (Reciprocal Rank Fusion) 融合两路排名

        对每条文档，RRF 分数 = SUM(1 / (k + rank_i))
        rank_i 是文档在各路中的排名，k 是平滑参数
        """
        scores: Dict[str, dict] = {}  # doc_id → {text, rrf_score}

        # 稠密路贡献（乘以 dense_weight 降低噪声影响）
        w_dense = self._dense_weight
        for rank, doc in enumerate(dense_docs, start=1):
            doc_id = doc["id"]
            rrf = w_dense / (RRF_K + rank)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += rrf
            else:
                scores[doc_id] = {"text": doc["text"], "rrf_score": rrf}

        # BM25 路贡献
        for rank, doc in enumerate(bm25_docs, start=1):
            doc_id = doc["id"]
            rrf = 1.0 / (RRF_K + rank)
            if doc_id in scores:
                scores[doc_id]["rrf_score"] += rrf
            else:
                scores[doc_id] = {"text": doc["text"], "rrf_score": rrf}

        # 按 RRF 分数降序
        ranked = sorted(scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return [
            {"id": doc_id, "text": info["text"], "score": info["rrf_score"]}
            for doc_id, info in ranked[:top_k]
        ]
