"""Chroma 向量库封装：存储 + 语义检索 + metadata 过滤 + source 加权"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

import chromadb

logger = logging.getLogger(__name__)


class ChromaStore:
    """Chroma 向量库封装

    生产环境：注入百炼 EmbeddingFunction
    测试环境：注入 HashEmbedding
    """

    def __init__(
        self,
        collection_name: str,
        embedding_fn,
        persist_dir: Optional[str] = None,
    ):
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        # 先试获取已有 collection，没有则创建
        try:
            self._collection = self._client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn,
            )
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    def add(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
    ) -> None:
        """批量添加文档"""
        if not ids:
            return
        self._collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas or [{}] * len(ids),
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_meta: Optional[dict] = None,
        source_weights: Optional[Dict[str, float]] = None,
    ) -> List[dict]:
        """稠密向量检索 + metadata 过滤 + source 加权

        Returns: [{"id": ..., "text": ..., "score": ..., "source": ..., "category": ...}, ...]
        """
        if self._collection.count() == 0:
            return []

        where = None
        if filter_meta:
            where = {k: v for k, v in filter_meta.items() if v}

        # Chroma 检索取 top_k * 2 来做加权后重新排序
        fetch_k = top_k * 2
        results = self._collection.query(
            query_texts=[query],
            n_results=min(fetch_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] or {}
            distance = results["distances"][0][i]
            # cosine distance → score (0=distance=best → 1.0=best)
            score = 1.0 - distance / 2.0

            # source 加权
            if source_weights:
                src = meta.get("source", "")
                weight = source_weights.get(src, 1.0)
                score *= weight

            items.append({
                "id": doc_id,
                "text": results["documents"][0][i],
                "score": score,
                "source": meta.get("source", ""),
                "category": meta.get("category", ""),
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        return items[:top_k]
