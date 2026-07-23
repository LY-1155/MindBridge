"""Query 分类器 TDD 测试"""

import pytest
from core.rag.query_classifier import QueryClassifier, QueryClassification


class FakeLLM:
    """Fake LLM，返回预设的 JSON 分类结果"""

    def __init__(self, classification: dict):
        self._classification = classification
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        from langchain_core.messages import AIMessage
        import json
        return AIMessage(content=json.dumps(self._classification, ensure_ascii=False))


class TestQueryClassifier:
    """Query 分类器：LLM 分类查询 → source + categories"""

    def test_classifies_medication_query_as_public(self):
        """药物相关查询 → source=public, category=medication_knowledge"""
        clf = QueryClassifier(llm=FakeLLM({
            "source": "public",
            "categories": ["medication_knowledge"]
        }))

        result = clf.classify("舍曲林的副作用是什么")

        assert isinstance(result, QueryClassification)
        assert result.source == "public"
        assert "medication_knowledge" in result.categories

    def test_classifies_therapy_query_as_public(self):
        """疗法技术查询 → source=public, category=therapy_techniques"""
        clf = QueryClassifier(llm=FakeLLM({
            "source": "public",
            "categories": ["therapy_techniques", "disorder_knowledge"]
        }))

        result = clf.classify("CBT 怎么处理焦虑")

        assert result.source == "public"
        assert "therapy_techniques" in result.categories

    def test_classifies_clinical_experience_as_private(self):
        """临床经验类查询 → source=private"""
        clf = QueryClassifier(llm=FakeLLM({
            "source": "private",
            "categories": ["clinical"]
        }))

        result = clf.classify("如何回应来访者对家庭的抱怨")

        assert result.source == "private"
        assert "clinical" in result.categories

    def test_injects_query_into_llm_prompt(self):
        """验证用户查询被注入到 LLM prompt"""
        clf = QueryClassifier(llm=FakeLLM({
            "source": "public",
            "categories": ["therapy_techniques"]
        }))

        clf.classify("森田疗法对强迫症有效吗")

        prompt = str(clf._llm.last_messages[-1].content)  # noqa: SLF001
        assert "森田疗法" in prompt
        assert "强迫症" in prompt

    def test_returns_both_source_on_llm_parse_failure(self):
        """LLM 输出无法解析时，默认返回 source=all"""
        bad_llm = FakeLLM("不是有效的 JSON 输出")

        clf = QueryClassifier(llm=bad_llm)
        result = clf.classify("随便问个问题")

        # 回退策略：返回 source=all, categories=[]，表示不过滤
        assert result.source == "all"
        assert result.categories == []

    def test_cross_domain_query_returns_both_sources(self):
        """跨域查询：私有临床经验 + 公有疗法知识都需要"""
        clf = QueryClassifier(llm=FakeLLM({
            "source": "both",
            "categories": ["clinical", "therapy_techniques"]
        }))

        result = clf.classify("来访者焦虑时躯体化严重，用什么干预技术合适")

        assert result.source == "both"
        assert len(result.categories) >= 2
