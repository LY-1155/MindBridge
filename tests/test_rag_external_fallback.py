"""回归测试：外部 API fallback（Tavily 药物补充）在生产路径不再死代码（A2 修复）

修复前：外部兜底只在 HybridRetriever.retrieve() 里调用，但生产走 rerank 分支
（retrieve_rrf_with_ids）不经它；非 rerank 分支里 supplement 也被 candidates[:final_k]
截断——外部兜底在生产路径从未真正生效。
修复后：收敛到 KnowledgeRetriever.retrieve()（生产入口），重排/非重排分支都覆盖。
"""

from unittest.mock import MagicMock

import pytest

from core.rag.bm25_index import BM25Index
from core.rag.hybrid_retriever import HybridRetriever
from modules.intervention.rag.retriever import KnowledgeRetriever


class _BoomExternal:
    """一旦被调用就失败——用于断言某路径不应触发外部搜索"""

    def search(self, query, retrieved_texts):
        raise AssertionError("外部搜索不应在此路径被触发")


class _FakeExternal:
    def __init__(self, supplements):
        self._supplements = supplements
        self.calls = []

    def search(self, query, retrieved_texts):
        self.calls.append((query, retrieved_texts))
        return list(self._supplements)


class _PassthroughReranker:
    _top_n = 5

    def rerank(self, query, docs):
        return docs


def _knowledge_retriever(external, reranker=None) -> KnowledgeRetriever:
    kr = KnowledgeRetriever.__new__(KnowledgeRetriever)
    kr._retrieval_config = {"top_k": 3}
    kr._rewriter = None
    kr._reranker = reranker
    kr._external = external

    bm25 = BM25Index()
    bm25.add("d1", "认知行为疗法是治疗焦虑的有效方法", {"source": "public", "category": "coping_strategies"})
    kr._bm25 = bm25

    # Chroma 不可用 → hybrid 内部自动降级 BM25-only；不触发 HTTP/持久化连接
    kr._hybrid = HybridRetriever(
        chroma_store=None,
        bm25_index=bm25,
        external_retriever=external,
    )
    return kr


def test_external_fallback_fires_on_rerank_production_path():
    """生产形状（带 reranker）：外部补充应被检索并保留在 top-k 里"""
    ext = _FakeExternal(["【舍曲林 - 实时搜索补充】\n以下信息由 Tavily 实时检索..."])
    kr = _knowledge_retriever(ext, reranker=_PassthroughReranker())
    kr._ensure_hybrid = MagicMock()  # 跳过真实 Chroma 连接

    results = kr.retrieve("舍曲林 有什么副作用", top_k=3)

    assert any("实时搜索补充" in r for r in results), "外部补充应进入最终结果"
    assert len(ext.calls) == 1, "外部搜索应恰好触发一次"


def test_external_fallback_fires_on_non_rerank_path():
    """无 reranker：外部补充同样应生效（不再被截断丢弃）"""
    ext = _FakeExternal(["【舍曲林 - 实时搜索补充】..."])
    kr = _knowledge_retriever(ext, reranker=None)
    kr._ensure_hybrid = MagicMock()

    results = kr.retrieve("舍曲林 副作用", top_k=3)

    assert any("实时搜索补充" in r for r in results)
    assert len(ext.calls) == 1, "非 rerank 分支不应重复触发外部搜索"


def test_hybrid_retrieve_no_longer_calls_external():
    """HybridRetriever.retrieve 不再自己调外部（已收敛到 KnowledgeRetriever）"""
    bm25 = BM25Index()
    bm25.add("d1", "焦虑 认知行为疗法 失眠", {"source": "public", "category": "coping_strategies"})
    hybrid = HybridRetriever(
        chroma_store=None,
        bm25_index=bm25,
        external_retriever=_BoomExternal(),
    )

    results = hybrid.retrieve("焦虑", top_k=3)

    assert results, "本地 BM25 结果应正常返回"
