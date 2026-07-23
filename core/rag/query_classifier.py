"""Query 分类器：LLM 分析用户查询，产出 source + categories 用于 metadata 过滤"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX

logger = logging.getLogger(__name__)

CLASSIFIER_SYSTEM_PROMPT = """你是一个心理学知识库查询分类器。分析用户的问题，判断需要检索的知识来源和类别。

## 知识来源 (source)
- "private": 临床咨询经验、案例分析、咨询师技法——category=clinical
- "public": 通用心理学知识（药物、诊断、应对策略、睡眠、创伤、哀伤、人际关系）
- "both": 同时需要私有和公有知识
- "all": 不确定，不过滤

## 知识类别 (categories) — 可多选

私有类别：
- "clinical": 临床咨询技法、家庭治疗、个案概念化、沟通分析、互动循环打破

公有类别：
- "disorder_knowledge": 心理异常症状、诊断标准、病程、鉴别诊断
- "coping_strategies": 治疗技术原理 + 可操作应对技巧（CBT/ACT/DBT/正念/放松训练/接地技巧等）
- "medication_knowledge": 精神科药物（副作用、剂量、禁忌症、停药反应）
- "sleep_health": 睡眠健康、失眠干预、CBT-I、睡眠卫生、昼夜节律
- "trauma_and_stress": 创伤知情照护、心理急救、安全稳定化、解离应对
- "grief_and_loss": 哀伤理论、丧失应对、居丧支持、延长哀伤
- "relationships": 人际关系、亲密关系、依恋风格、沟通模式、冲突解决
- "psychology_basics": 心理学基础概念（情绪调节、认知偏差、发展心理学常识等）

## 判断指南
- 涉及药物名、副作用、剂量 → medication_knowledge
- 涉及睡不着、失眠、睡眠质量 → sleep_health
- 涉及创伤事件、PTSD、闪回、解离 → trauma_and_stress
- 涉及丧亲、分手、丧失、走不出来 → grief_and_loss
- 涉及人际冲突、亲密关系、依恋 → relationships
- 涉及症状诊断、DSM、病程 → disorder_knowledge
- 涉及具体应对方法、放松技巧、疗法操作 → coping_strategies
- 涉及咨询师视角的技法、家庭治疗经验 → clinical (private)
- 模糊的基础心理学概念 → psychology_basics

## 输出格式
严格输出 JSON，不要任何其他文字：
{"source": "public", "categories": ["medication_knowledge"]}

## 示例
Q: "舍曲林的副作用是什么"
A: {"source": "public", "categories": ["medication_knowledge"]}

Q: "如何回应来访者的阻抗"
A: {"source": "private", "categories": ["clinical"]}

Q: "分手后一直走不出来怎么办"
A: {"source": "public", "categories": ["grief_and_loss", "coping_strategies"]}

Q: "失眠半年了，除了吃药还有什么办法"
A: {"source": "public", "categories": ["sleep_health", "coping_strategies", "medication_knowledge"]}

Q: "经历过家暴，总是做噩梦怎么办"
A: {"source": "public", "categories": ["trauma_and_stress", "sleep_health", "coping_strategies"]}"""


@dataclass
class QueryClassification:
    """查询分类结果

    最终 9 类精简分类（见 ADR 0007）：
      private: clinical
      public:  disorder_knowledge / coping_strategies / medication_knowledge /
               sleep_health / trauma_and_stress / grief_and_loss /
               relationships / psychology_basics
    """
    source: str = "all"          # "private" | "public" | "both" | "all"
    categories: List[str] = field(default_factory=list)


class QueryClassifier:
    """使用 LLM 对用户查询做来源和类别分类"""

    def __init__(self, llm):
        self._llm = llm

    def classify(self, query: str) -> QueryClassification:
        """分类用户查询，返回 QueryClassification"""
        try:
            wrapped = wrap_user_text(query)
            messages = [
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT + INSTRUCTION_HIERARCHY_SUFFIX),
                HumanMessage(content=wrapped),
            ]
            response = self._llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            result = self._parse(raw)
        except Exception:
            logger.warning("Query 分类失败，回退到不过滤", exc_info=True)
            result = QueryClassification(source="all", categories=[])

        return result

    def _parse(self, raw: str) -> QueryClassification:
        """解析 LLM JSON 输出"""
        raw = raw.strip()
        # 尝试直接解析
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 JSON 块
            import re
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    raise
            else:
                raise

        return QueryClassification(
            source=data.get("source", "all"),
            categories=data.get("categories", []),
        )
