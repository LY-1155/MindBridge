"""回归测试：Embedding 失败不再静默返回零向量（A1 修复）

修复前：QianwenEmbedding/BGEM3Embedding 在 API/推理失败时返回全零向量，
上游拿零向量做余弦相似度 → 检索"成功"地返回随机相似度的垃圾结果。
修复后：抛 EmbeddingError，HybridRetriever 捕获并降级为 BM25-only。
"""

from unittest.mock import MagicMock

import pytest

from core.rag.bm25_index import BM25Index
from core.rag.embedder import BGEM3Embedding, EmbeddingError, QianwenEmbedding
from core.rag.hybrid_retriever import HybridRetriever


def _qianwen_with_client(client) -> QianwenEmbedding:
    emb = QianwenEmbedding.__new__(QianwenEmbedding)
    emb._client = client
    emb._model = "test-model"
    emb._batch_size = 2
    emb._dimensions = None
    emb._dim = None
    return emb


def test_qianwen_api_failure_raises_embedding_error():
    client = MagicMock()
    client.embeddings.create.side_effect = RuntimeError("connection refused")
    emb = _qianwen_with_client(client)
    with pytest.raises(EmbeddingError, match="connection refused"):
        emb(["你好", "世界"])


def test_qianwen_empty_result_raises_embedding_error():
    client = MagicMock()
    client.embeddings.create.return_value = MagicMock(data=[])
    emb = _qianwen_with_client(client)
    with pytest.raises(EmbeddingError, match="空结果"):
        emb(["你好"])


def test_bge_inference_failure_raises_embedding_error():
    emb = BGEM3Embedding.__new__(BGEM3Embedding)
    emb._model = MagicMock()
    emb._model.encode.side_effect = RuntimeError("cuda oom")
    emb._batch_size = 2
    emb._dim = 1024
    with pytest.raises(EmbeddingError, match="cuda oom"):
        emb(["你好"])


def _hybrid_with_failing_chroma() -> HybridRetriever:
    bm25 = BM25Index()
    bm25.add("d1", "失眠 睡眠障碍 入睡困难", {"source": "public", "category": "sleep_health"})
    bm25.add("d2", "焦虑 紧张 恐慌", {"source": "public", "category": "coping_strategies"})

    class FailingChroma:
        def search(self, *args, **kwargs):
            raise EmbeddingError("Embedding API 调用失败: boom")

    return HybridRetriever(chroma_store=FailingChroma(), bm25_index=bm25)


def test_hybrid_degrades_to_bm25_on_embedding_failure():
    """稠密路 Embedding 失败 → RRF 融合自然只剩 BM25 结果，不抛异常不返垃圾"""
    hybrid = _hybrid_with_failing_chroma()
    docs = hybrid.retrieve_rrf_with_ids("睡眠不好", top_k=15)
    assert docs, "降级后不应为空"
    assert all("失眠" in d["text"] for d in docs), "降级后应命中 BM25 的睡眠文档"


def test_hybrid_degrades_to_bm25_when_chroma_none():
    bm25 = BM25Index()
    bm25.add("d1", "失眠 睡眠障碍", {"source": "public", "category": "sleep_health"})
    hybrid = HybridRetriever(chroma_store=None, bm25_index=bm25)
    docs = hybrid.retrieve_rrf_with_ids("睡眠", top_k=15)
    assert docs, "Chroma 不可用时也应降级到 BM25"
