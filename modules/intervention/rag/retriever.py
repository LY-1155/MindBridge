"""知识库检索器 v3：统一索引 + 混合检索 + 外部 API fallback

旧版为三层逐级回退（私有一公有—外部 API），详见 ADR-0006。
v3 流程：Query Rewriting → Chroma 稠密 + BM25 关键词 → RRF 融合 → 外部 API fallback → top-3 文本
v3.1 (2026-07): 新增可选 QueryRewriter，检索前自动补充心理学专业术语。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.rag.bm25_index import BM25Index
from core.rag.chroma_store import ChromaStore
from core.rag.hybrid_retriever import HybridRetriever
from core.rag.external_retriever import ExternalRetriever
from core.rag.query_rewriter import QueryRewriter

logger = logging.getLogger(__name__)

_DEFAULT_SOURCES_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "knowledge" / "sources.json"
)
_CHROMA_PERSIST_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "knowledge" / "chroma_index"
)


def _load_sources(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _DEFAULT_SOURCES_PATH
    if not p.exists():
        logger.warning("知识库配置文件不存在: %s", p)
        return {"layers": [], "retrieval": {"top_k": 3, "min_score": 0.5}}
    return json.loads(p.read_text(encoding="utf-8"))


def _build_bm25_from_jsonl(sources_path: Optional[Path] = None) -> BM25Index:
    """从 JSONL 数据构建 BM25 索引"""
    config = _load_sources(sources_path)
    bm25 = BM25Index()

    for layer in config.get("layers", []):
        if not layer.get("enabled") or layer.get("type") != "local":
            continue

        source_label = "private" if "private" in layer["id"] else "public"
        data_dir = Path(layer["path"])
        if not data_dir.is_absolute():
            data_dir = (
                Path(__file__).resolve().parent.parent.parent.parent / data_dir
            )

        if not data_dir.exists():
            continue

        for f in sorted(data_dir.glob("*.jsonl")):
            category = _category_from_filename(f.name)
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = doc.get("id", f"{source_label}_{len(bm25._docs)}")  # noqa: SLF001
                text = f"{doc.get('title', '')} | {' '.join(doc.get('tags', []))} | {doc.get('content', '')}"
                bm25.add(doc_id, text, {"source": source_label, "category": category})

    logger.info("BM25 索引构建完成：%d 条知识", len(bm25._docs))  # noqa: SLF001
    return bm25


def _category_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for suffix in ["_knowledge", "_info", "_data"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


class KnowledgeRetriever:
    """知识库检索器 v3.2

    接口保持 v1 兼容：retriever.retrieve(query, top_k=3) → List[str]
    内部：QueryRewriter → BM25 粗检索 → LLMReranker 精排

    改写器与重排器默认关闭。注入实例即可启用。
    """

    def __init__(
        self,
        sources_path: Optional[Path] = None,
        rewriter: Optional[QueryRewriter] = None,
        reranker=None,  # Optional[LLMReranker]
    ):
        """
        Args:
            sources_path: sources.json 路径
            rewriter: 可选的查询改写器。传入 None 则跳过改写步骤。
            reranker: 可选的 LLM 重排器。传入 None 则直接返回粗检索结果。
        """
        config = _load_sources(sources_path)
        self._config = config
        self._retrieval_config = config.get("retrieval", {"top_k": 3, "min_score": 0.5})
        self._source_weights = config.get("source_weights", {"private": 1.2, "public": 1.0})

        # 查询改写器（可选，默认关闭）
        self._rewriter = rewriter
        # LLM 重排器（可选，默认关闭）
        self._reranker = reranker

        # 构建 BM25 索引
        self._bm25 = _build_bm25_from_jsonl(sources_path)

        # Chroma 连接（延迟初始化，避免无数据时报错）
        self._chroma: Optional[ChromaStore] = None
        self._hybrid: Optional[HybridRetriever] = None

        # 外部 API 检索器（从 sources.json external_api layer 加载）
        external_config = None
        for layer in config.get("layers", []):
            if layer.get("id") == "external_api":
                external_config = layer
                break
        self._external = ExternalRetriever(external_config)

    def _ensure_hybrid(self):
        """延迟初始化 HybridRetriever"""
        if self._hybrid is not None:
            return

        # Chroma
        if os.path.isdir(str(_CHROMA_PERSIST_DIR)):
            try:
                from core.rag.embedder import create_embedder
                from config.settings import settings

                embedder = create_embedder(backend=settings.EMBEDDING_BACKEND)
                self._chroma = ChromaStore(
                    collection_name="knowledge_base",
                    embedding_fn=embedder,
                    persist_dir=str(_CHROMA_PERSIST_DIR),
                )
            except Exception:
                logger.warning("Chroma 索引加载失败，将仅使用 BM25 检索", exc_info=True)
                self._chroma = None
        else:
            logger.info("Chroma 索引目录不存在，将仅使用 BM25 检索")

        from config.settings import settings

        self._hybrid = HybridRetriever(
            chroma_store=self._chroma,
            bm25_index=self._bm25,
            source_weights=self._source_weights,
            external_retriever=self._external,
            dense_weight=settings.RRF_DENSE_WEIGHT,
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """主检索入口：返回文本片段列表（兼容原接口）

        流程：Query Rewriting → Union(Chroma + BM25) → Reranker 精排 → top-k
        """
        final_k = top_k or self._retrieval_config.get("top_k", 3)

        # Step 0: 查询改写（可选）
        search_query = query
        if self._rewriter is not None:
            search_query = self._rewriter.rewrite(query)

        # Step 1: 粗检索
        candidates: List[str] = []
        try:
            self._ensure_hybrid()

            if self._reranker is not None:
                # ── 有 Reranker：Union 模式 ──
                # 两路各取 top_n 条，去重合并后交给 Reranker 精排
                fetch_k = self._reranker._top_n  # noqa: SLF001
                union_docs = self._hybrid.retrieve_union_with_ids(
                    search_query, top_k=fetch_k
                )
                candidates = [d["text"] for d in union_docs]
            else:
                # ── 无 Reranker：RRF 模式（兼容旧行为） ──
                candidates = self._hybrid.retrieve(
                    search_query, top_k=final_k
                )
        except Exception:
            logger.warning("混合检索失败，回退到 BM25 关键词检索", exc_info=True)
            fetch_k = self._reranker._top_n if self._reranker else final_k  # noqa: SLF001
            candidates = self._fallback_bm25(search_query, fetch_k)

        # Step 2: 精排（可选）
        if self._reranker is not None and len(candidates) > final_k:
            candidates = self._reranker.rerank(query, candidates)

        candidates = candidates[:final_k]

        if candidates:
            logger.info("知识库检索命中: query=%.50s..., hits=%d", query, len(candidates))
        else:
            logger.info("知识库检索未命中: query=%.50s...", query)

        return candidates

    def _fallback_bm25(self, query: str, top_k: int) -> List[str]:
        """BM25 关键词检索兜底"""
        raw = self._bm25.search(query, top_k=top_k)
        if not raw:
            return []
        texts = []
        for doc_id, _ in raw:
            text = self._bm25._docs.get(doc_id, "")  # noqa: SLF001
            if text:
                texts.append(text)
        return texts[:top_k]


# 模块级单例
_retriever: Optional[KnowledgeRetriever] = None


def get_knowledge_retriever(
    sources_path: Optional[Path] = None,
    rewriter: Optional[QueryRewriter] = None,
) -> KnowledgeRetriever:
    """获取 KnowledgeRetriever 单例。

    首次调用时创建实例。传入 rewriter 启用查询改写；
    传入 None（默认）则跳过改写步骤，行为与 v3 完全一致。
    """
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever(
            sources_path=sources_path,
            rewriter=rewriter,
        )
    return _retriever
