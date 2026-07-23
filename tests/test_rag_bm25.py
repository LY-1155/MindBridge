"""BM25 关键词检索 TDD 测试"""

import pytest
from core.rag.bm25_index import BM25Index


class TestBM25Index:
    """BM25 索引：jieba 分词 + 关键词检索 + metadata 过滤"""

    @pytest.fixture
    def index(self):
        idx = BM25Index()
        idx.add("doc_1", "认知行为疗法是治疗焦虑的有效方法", {"source": "public", "category": "therapy_techniques"})
        idx.add("doc_2", "失眠的认知行为干预包括刺激控制和睡眠限制", {"source": "public", "category": "disorder_knowledge"})
        idx.add("doc_3", "正念冥想可以降低焦虑水平", {"source": "public", "category": "therapy_techniques"})
        idx.add("doc_4", "家庭治疗中需识别隐性沟通与三角动力", {"source": "private", "category": "clinical"})
        return idx

    def test_exact_keyword_match_returns_relevant_docs(self, index):
        """精确关键词匹配：搜"焦虑"应返回含焦虑的文档"""
        results = index.search("焦虑", top_k=3)

        assert len(results) > 0
        doc_ids = [r[0] for r in results]
        assert "doc_1" in doc_ids or "doc_3" in doc_ids  # 至少命中一个含焦虑的

    def test_multi_word_query_matches_better_than_single_word(self, index):
        """多词查询比单词查询排序更合理：搜"CBT 认知行为"应把认知行为疗法排前面"""
        results = index.search("认知行为疗法 焦虑", top_k=3)

        assert len(results) > 0
        # doc_1 含完整的"认知行为疗法"和"焦虑"，得分应最高
        assert results[0][0] == "doc_1"

    def test_empty_query_returns_empty(self, index):
        """空查询返回空列表"""
        results = index.search("", top_k=3)
        assert results == []

    def test_filter_by_source(self, index):
        """按 source 过滤：只搜 private 应只返回私有库文档"""
        results = index.search("家庭治疗 隐性沟通 三角动力", top_k=3, filter_meta={"source": "private"})

        doc_ids = [r[0] for r in results]
        assert "doc_4" in doc_ids
        assert "doc_1" not in doc_ids  # public 被过滤

    def test_filter_by_category(self, index):
        """按 category 过滤：只搜 therapy_techniques"""
        results = index.search("焦虑", top_k=3, filter_meta={"category": "therapy_techniques"})

        doc_ids = [r[0] for r in results]
        assert "doc_1" in doc_ids or "doc_3" in doc_ids
        assert "doc_2" not in doc_ids  # disorder_knowledge 被过滤

    def test_returns_scores_in_descending_order(self, index):
        """返回结果按得分降序排列"""
        results = index.search("焦虑 认知", top_k=5)

        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_without_metadata(self):
        """无 metadata 也能正常添加和检索"""
        idx = BM25Index()
        idx.add("doc_x", "渐进式肌肉放松法", metadata=None)

        results = idx.search("肌肉放松", top_k=3)
        assert len(results) == 1
        assert results[0][0] == "doc_x"

    def test_filter_multiple_conditions(self, index):
        """多个过滤条件：source + category 同时过滤
        搜"焦虑 认知 治疗"，过滤 source=public + category=therapy_techniques
        doc_1（认知行为疗法、焦虑）和 doc_3（焦虑）都匹配，doc_2（disorder_knowledge）被过滤
        """
        results = index.search("焦虑 认知 治疗", top_k=3, filter_meta={
            "source": "public",
            "category": "therapy_techniques"
        })

        doc_ids = [r[0] for r in results]
        assert "doc_1" in doc_ids
        assert "doc_2" not in doc_ids  # disorder_knowledge 被过滤
        assert "doc_3" in doc_ids      # 含"焦虑"，且是 therapy_techniques
