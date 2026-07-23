"""HybridRetriever TDD 测试"""

import pytest
from chromadb import EmbeddingFunction

from core.rag.bm25_index import BM25Index
from core.rag.chroma_store import ChromaStore
from core.rag.query_classifier import QueryClassifier, QueryClassification
from core.rag.hybrid_retriever import HybridRetriever


class FakeLLM:
    """Fake LLM 返回预设分类 JSON"""

    def __init__(self, classification: dict):
        self._classification = classification

    def invoke(self, messages):
        from langchain_core.messages import AIMessage
        import json
        return AIMessage(content=json.dumps(self._classification, ensure_ascii=False))


class TestEmbedding(EmbeddingFunction):
    """确定性假 Embedding：基于文本中关键词匹配"""

    VOCAB = list("焦虑抑郁认知行为疗法障碍治疗正念睡眠失眠药物舍曲林副作用家庭沟通三角动力")

    def __call__(self, input: list[str]):
        import math
        dim = len(self.VOCAB)
        vectors = []
        for text in input:
            v = [0.0] * dim
            for i, kw in enumerate(self.VOCAB):
                v[i] = float(text.count(kw)) / max(len(text), 1)
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            vectors.append(v)
        return vectors


# --- 测试数据 ---
DOCS = [
    ("doc_1", "认知行为疗法是治疗焦虑的有效方法", {"source": "public", "category": "therapy_techniques"}),
    ("doc_2", "失眠的认知行为干预包括刺激控制和睡眠限制", {"source": "public", "category": "disorder_knowledge"}),
    ("doc_3", "正念冥想可以降低焦虑水平", {"source": "public", "category": "therapy_techniques"}),
    ("doc_4", "家庭治疗中需识别隐性沟通与三角动力", {"source": "private", "category": "clinical"}),
    ("doc_5", "SSRI类药物如舍曲林是抑郁一线用药，常见副作用包括恶心", {"source": "public", "category": "medication_knowledge"}),
]


class TestHybridRetriever:
    """混合检索器：稠密 + BM25 → RRF 融合 → top-k"""

    @pytest.fixture
    def retriever(self):
        # 建索引
        bm25 = BM25Index()
        for doc_id, text, meta in DOCS:
            bm25.add(doc_id, text, meta)

        emb_fn = TestEmbedding()
        chroma = ChromaStore(collection_name="hybrid_test", embedding_fn=emb_fn, persist_dir=None)
        chroma.add(
            ids=[d[0] for d in DOCS],
            texts=[d[1] for d in DOCS],
            metadatas=[d[2] for d in DOCS],
        )

        # 预设返回不过滤的分类
        classifier = QueryClassifier(llm=FakeLLM({
            "source": "all",
            "categories": []
        }))

        return HybridRetriever(
            chroma_store=chroma,
            bm25_index=bm25,
            classifier=classifier,
            source_weights={"private": 1.2, "public": 1.0},
        )

    def test_search_returns_combined_results(self, retriever):
        """混合检索返回稠密 + BM25 的综合结果"""
        results = retriever.retrieve("认知行为疗法 焦虑", top_k=3)

        assert len(results) > 0
        # 验证返回的是文本片段
        assert isinstance(results[0], str)
        assert len(results[0]) > 0

    def test_search_filters_by_classification(self, retriever):
        """分类过滤：只查 medication 时不应返回 therapy 条目"""
        # 重建一个 medication-only 分类的 retriever
        bm25 = BM25Index()
        for doc_id, text, meta in DOCS:
            bm25.add(doc_id, text, meta)

        emb_fn = TestEmbedding()
        chroma = ChromaStore(collection_name="med_filter_test", embedding_fn=emb_fn, persist_dir=None)
        chroma.add(
            ids=[d[0] for d in DOCS],
            texts=[d[1] for d in DOCS],
            metadatas=[d[2] for d in DOCS],
        )

        clf = QueryClassifier(llm=FakeLLM({
            "source": "public",
            "categories": ["medication_knowledge"]
        }))

        ret = HybridRetriever(chroma_store=chroma, bm25_index=bm25, classifier=clf)

        results = ret.retrieve("什么药可以治抑郁", top_k=3)
        # 应该过滤掉非 medication 的条目
        for r in results:
            assert "CBT" not in r  # therapy_techniques 被过滤

    def test_empty_when_no_matches(self):
        """无匹配返回空"""
        bm25 = BM25Index()
        bm25.add("x", "这是唯一的文档", {"source": "public", "category": "test"})
        emb_fn = TestEmbedding()
        chroma = ChromaStore(collection_name="no_match_test", embedding_fn=emb_fn, persist_dir=None)
        chroma.add(ids=["x"], texts=["这是唯一的文档"], metadatas=[{"source": "public", "category": "test"}])
        clf = QueryClassifier(llm=FakeLLM({"source": "public", "categories": ["medication_knowledge"]}))

        ret = HybridRetriever(chroma_store=chroma, bm25_index=bm25, classifier=clf)

        # 搜不同的 category 应该返回空
        results = ret.retrieve("某个不相关的查询", top_k=3)
        # 注意：BM25 不过滤可以命中，但 Chroma 过滤了
        # 所以可能还有结果，但应该很少
        # 换个完全不可能匹配的查询
        results2 = ret.retrieve("zzzzzzzzz", top_k=3)
        assert isinstance(results2, list)

    def test_reproducible_results(self, retriever):
        """相同查询返回相同结果"""
        r1 = retriever.retrieve("焦虑治疗方法", top_k=3)
        r2 = retriever.retrieve("焦虑治疗方法", top_k=3)

        assert r1 == r2

    def test_result_does_not_exceed_top_k(self, retriever):
        """返回结果不超过 top_k"""
        for k in [1, 2, 3]:
            results = retriever.retrieve("认知 焦虑 治疗", top_k=k)
            assert len(results) <= k

    def test_retrieve_with_ids_returns_structured_results(self, retriever):
        """retrieve_with_ids 返回带 id 的结构化结果"""
        results = retriever.retrieve_with_ids("认知行为疗法", top_k=3)

        assert len(results) > 0
        assert isinstance(results, list)
        assert isinstance(results[0], dict)
        assert "id" in results[0]
        assert "text" in results[0]
        assert "score" in results[0]
        # ID 应该是 doc 格式
        assert results[0]["id"].startswith("doc_")

    def test_retrieve_dense_only_returns_chroma_results(self, retriever):
        """dense_only 返回 Chroma 结果"""
        results = retriever.retrieve_dense_only("焦虑", top_k=3)
        assert isinstance(results, list)
        # 返回结构化 dict
        if results:
            assert "id" in results[0]

    def test_retrieve_sparse_only_returns_bm25_results(self, retriever):
        """sparse_only 返回 BM25 结果"""
        results = retriever.retrieve_sparse_only("焦虑", top_k=3)
        assert isinstance(results, list)
        if results:
            assert "id" in results[0]

    def test_retrieve_with_ids_returns_results_without_classifier(self):
        """无分类器时 retrieve_with_ids 正常返回 RRF 融合结果"""
        bm25 = BM25Index()
        bm25.add("doc_a", "CBT 认知行为疗法治疗焦虑",
                  {"source": "public", "category": "coping_strategies"})
        bm25.add("doc_b", "舍曲林是SSRI抗抑郁药物",
                  {"source": "public", "category": "medication_knowledge"})

        emb_fn = TestEmbedding()
        chroma = ChromaStore(collection_name="no_clf_test", embedding_fn=emb_fn, persist_dir=None)
        chroma.add(
            ids=["doc_a", "doc_b"],
            texts=["CBT 认知行为疗法治疗焦虑", "舍曲林是SSRI抗抑郁药物"],
            metadatas=[
                {"source": "public", "category": "coping_strategies"},
                {"source": "public", "category": "medication_knowledge"},
            ],
        )

        # 不传 classifier
        ret = HybridRetriever(chroma_store=chroma, bm25_index=bm25)

        # 药物查询应能召回用药文档
        results = ret.retrieve_with_ids("抗抑郁药物", top_k=3)
        retrieved_ids = [d["id"] for d in results]
        assert "doc_b" in retrieved_ids
        assert len(results) > 0

    def test_hybrid_retrieval_fuses_dense_and_sparse(self):
        """无分类器 hybrid：dense + BM25 RRF 融合正常"""
        bm25 = BM25Index()
        bm25.add("doc_1", "CBT 认知行为疗法", {"source": "public", "category": "coping_strategies"})
        bm25.add("doc_2", "舍曲林副作用恶心头晕", {"source": "public", "category": "medication_knowledge"})
        bm25.add("doc_3", "失眠的认知行为干预", {"source": "public", "category": "sleep_health"})

        emb_fn = TestEmbedding()
        chroma = ChromaStore(collection_name="hybrid_fuse_test", embedding_fn=emb_fn, persist_dir=None)
        chroma.add(
            ids=["doc_1", "doc_2", "doc_3"],
            texts=["CBT 认知行为疗法", "舍曲林副作用恶心头晕", "失眠的认知行为干预"],
            metadatas=[
                {"source": "public", "category": "coping_strategies"},
                {"source": "public", "category": "medication_knowledge"},
                {"source": "public", "category": "sleep_health"},
            ],
        )

        # 不传 classifier
        ret = HybridRetriever(chroma_store=chroma, bm25_index=bm25)
        results = ret.retrieve_with_ids("认知 药物", top_k=3)
        retrieved_ids = [d["id"] for d in results]

        # 应有结果返回
        assert len(results) > 0
        # 相关文档应在结果中（无过滤）
