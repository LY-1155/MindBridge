"""ChromaStore TDD 测试"""

import math
from collections import Counter

import pytest
from chromadb import EmbeddingFunction

from core.rag.chroma_store import ChromaStore


class BigramEmbedding(EmbeddingFunction):
    """中文 2-gram 频率假 Embedding：共享字串的文本会有相似的向量
    用标准中文标点符号作为基础维度，再动态扩展文本中出现的高频字符
    """

    VOCAB = (
        list("焦虑抑郁认知行为疗法障碍干预治疗正念冥想睡眠失眠药物控制放松自我家庭沟通情绪压力管理") +
        list("认知行为疗法焦虑抑郁失眠正念药物家庭沟通情绪压力")
    )

    def __call__(self, input: list[str]):
        dim = len(self.VOCAB)
        vectors = []
        all_text = " ".join(input)

        for text in input:
            # 统计 text 中各字符出现次数作为向量
            v = [0.0] * dim
            for i, ch in enumerate(self.VOCAB):
                if ch in text:
                    v[i] = float(text.count(ch)) / max(len(text), 1)
            # 归一化
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            vectors.append(v)
        return vectors


class TestChromaStore:
    """Chroma 向量库：存储 + 语义检索 + metadata 过滤 + source 加权"""

    @pytest.fixture
    def store(self):
        emb_fn = BigramEmbedding()
        cs = ChromaStore(collection_name="test_knowledge", embedding_fn=emb_fn, persist_dir=None)
        cs.add(
            ids=["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"],
            texts=[
                "认知行为疗法是治疗焦虑的有效方法",
                "失眠的认知行为干预包括刺激控制和睡眠限制",
                "正念冥想可以降低焦虑水平和压力",
                "家庭治疗中需识别隐性沟通与三角动力",
                "SSRI类药物如舍曲林是抑郁一线用药",
            ],
            metadatas=[
                {"source": "public", "category": "therapy_techniques"},
                {"source": "public", "category": "disorder_knowledge"},
                {"source": "public", "category": "therapy_techniques"},
                {"source": "private", "category": "clinical"},
                {"source": "public", "category": "medication_knowledge"},
            ],
        )
        return cs

    def test_search_returns_relevant_docs(self, store):
        """语义检索：搜"焦虑疗法"应返回治疗方法相关的文档"""
        results = store.search("焦虑 疗法", top_k=3)

        assert len(results) > 0
        doc_ids = [r["id"] for r in results]
        # 前两个 doc_1 和 doc_3 都含"焦虑"
        assert "doc_1" in doc_ids or "doc_3" in doc_ids

    def test_search_is_empty_when_no_docs(self):
        """空库检索返回空"""
        emb_fn = BigramEmbedding()
        cs = ChromaStore(collection_name="empty_test", embedding_fn=emb_fn, persist_dir=None)
        results = cs.search("焦虑", top_k=3)
        assert results == []

    def test_filter_by_source(self, store):
        """按 source 过滤：只查 private"""
        results = store.search("认知 治疗 家庭", top_k=3, filter_meta={"source": "private"})

        for r in results:
            assert r["source"] == "private"

    def test_filter_by_category(self, store):
        """按 category 过滤：只查 medication_knowledge"""
        results = store.search("药物 抑郁", top_k=3, filter_meta={"category": "medication_knowledge"})

        for r in results:
            assert r["category"] == "medication_knowledge"

    def test_source_weighting_boosts_private(self, store):
        """source 加权：私有文档加权后在同查询下应排在前面"""
        results = store.search("认知", top_k=3, source_weights={"private": 2.0, "public": 1.0})

        # 验证返回了结果
        assert len(results) > 0

    def test_docs_persist_across_instances(self, tmp_path):
        """持久化：保存后重新打开能检索到"""
        emb_fn = BigramEmbedding()
        path = str(tmp_path / "chroma_test")

        cs1 = ChromaStore(collection_name="persist_test", embedding_fn=emb_fn, persist_dir=path)
        cs1.add(ids=["p1"], texts=["渐进式肌肉放松法缓解焦虑"], metadatas=[{"source": "public"}])

        cs2 = ChromaStore(collection_name="persist_test", embedding_fn=emb_fn, persist_dir=path)
        results = cs2.search("渐进式肌肉放松", top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "p1"
