"""BM25 关键词检索索引：jieba 分词 + BM25 评分 + metadata 过滤"""

from __future__ import annotations

import math
from typing import List, Tuple, Dict, Optional

import jieba


class BM25Index:
    """轻量 BM25 索引：维持词频 + 文档元数据，不依赖外部数据库"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b

        # 文档存储
        self._docs: Dict[str, str] = {}         # doc_id → text
        self._meta: Dict[str, dict] = {}        # doc_id → metadata
        self._tokens: Dict[str, List[str]] = {} # doc_id → tokenized words

        # 倒排索引
        self._inv: Dict[str, Dict[str, int]] = {}  # term → {doc_id: count}

        # 惰性统计
        self._avgdl: float = 0.0
        self._doc_count: int = 0

    def add(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """添加文档到索引"""
        words = list(jieba.cut(text))
        words = [w.strip() for w in words if w.strip()]

        self._docs[doc_id] = text
        self._meta[doc_id] = metadata or {}
        self._tokens[doc_id] = words

        # 更新倒排索引
        tf: Dict[str, int] = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1

        for term, count in tf.items():
            if term not in self._inv:
                self._inv[term] = {}
            self._inv[term][doc_id] = count

        # 重算 avgdl
        total_words = sum(len(toks) for toks in self._tokens.values())
        self._doc_count = len(self._docs)
        self._avgdl = total_words / self._doc_count if self._doc_count > 0 else 0.0

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_meta: Optional[dict] = None,
    ) -> List[Tuple[str, float]]:
        """关键词检索，返回 [(doc_id, score), ...] 按得分降序"""
        if not query.strip():
            return []

        query_words = list(jieba.cut(query))
        query_words = [w.strip() for w in query_words if w.strip()]
        if not query_words:
            return []

        scores: Dict[str, float] = {}

        for qw in set(query_words):
            if qw not in self._inv:
                continue

            posting = self._inv[qw]
            nqi = len(posting)
            idf = math.log((self._doc_count - nqi + 0.5) / (nqi + 0.5) + 1.0)

            for doc_id, tf in posting.items():
                # metadata 过滤
                if filter_meta and not self._match_filter(doc_id, filter_meta):
                    continue

                dl = len(self._tokens[doc_id])
                score = idf * (tf * (self._k1 + 1)) / (tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def _match_filter(self, doc_id: str, filter_meta: dict) -> bool:
        """检查文档 metadata 是否满足所有过滤条件"""
        meta = self._meta.get(doc_id, {})
        for k, v in filter_meta.items():
            if meta.get(k) != v:
                return False
        return True
