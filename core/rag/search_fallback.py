"""搜索 fallback 模块：当知识库不覆盖某个精神科药物时，通过 Tavily 实时检索补充。

集成点：ExternalRetriever → KnowledgeRetriever.retrieve()（生产入口，收敛后调用）
三层架构中最外层的外部 API 兜底层。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Set

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 精神科药物识别表（覆盖中文通用名 + 英文）
# ---------------------------------------------------------------------------
# 此列表用于判断用户查询中是否提到精神科药物。包含：
#   1. 知识库已收录的 20 种药物
#   2. 临床常用但知识库未收录的药物（用于触发搜索 fallback）
# ---------------------------------------------------------------------------
_PSYCH_DRUG_KEYWORDS: Dict[str, Set[str]] = {
    # SSRI 抗抑郁药
    "舍曲林": {"sertraline", "左洛复"},
    "氟西汀": {"fluoxetine", "百忧解"},
    "帕罗西汀": {"paroxetine", "赛乐特"},
    "艾司西酞普兰": {"escitalopram", "来士普"},
    "西酞普兰": {"citalopram", "喜普妙"},
    "氟伏沙明": {"fluvoxamine", "兰释"},
    # SNRI 抗抑郁药
    "文拉法辛": {"venlafaxine", "怡诺思"},
    "度洛西汀": {"duloxetine", "欣百达"},
    # 其他抗抑郁药
    "米氮平": {"mirtazapine", "瑞美隆"},
    "安非他酮": {"bupropion", "悦刻"},
    "曲唑酮": {"trazodone", "美舒玉"},
    "伏硫西汀": {"vortioxetine", "心达悦"},
    "阿戈美拉汀": {"agomelatine", "维度新"},
    "圣约翰草": {"st john's wort", "路优泰"},
    # 非典型抗精神病药
    "奥氮平": {"olanzapine", "再普乐"},
    "喹硫平": {"quetiapine", "思瑞康"},
    "利培酮": {"risperidone", "维思通"},
    "阿立哌唑": {"aripiprazole", "安律凡"},
    "氯氮平": {"clozapine", "氯扎平"},
    "帕利哌酮": {"paliperidone", "芮达"},
    "齐拉西酮": {"ziprasidone", "卓乐定"},
    "氨磺必利": {"amisulpride", "索里昂"},
    # 第一代抗精神病药
    "氯丙嗪": {"chlorpromazine"},
    "氟哌啶醇": {"haloperidol"},
    "奋乃静": {"perphenazine"},
    "舒必利": {"sulpiride"},
    "五氟利多": {"penfluridol"},
    # 心境稳定剂
    "碳酸锂": {"lithium", "lithium carbonate"},
    "丙戊酸钠": {"sodium valproate", "valproate", "德巴金"},
    "丙戊酸镁": {"magnesium valproate"},
    "拉莫三嗪": {"lamotrigine", "利必通"},
    "卡马西平": {"carbamazepine", "得理多"},
    "奥卡西平": {"oxcarbazepine", "曲莱"},
    # 苯二氮䓬类
    "阿普唑仑": {"alprazolam", "佳静安定"},
    "氯硝西泮": {"clonazepam", "氯硝安定"},
    "劳拉西泮": {"lorazepam", "罗拉"},
    "地西泮": {"diazepam", "安定"},
    "艾司唑仑": {"estazolam", "舒乐安定"},
    "奥沙西泮": {"oxazepam", "舒宁"},
    "咪达唑仑": {"midazolam", "力月西"},
    # 其他抗焦虑药
    "丁螺环酮": {"buspirone", "布斯哌隆"},
    "坦度螺酮": {"tandospirone", "希德"},
    "羟嗪": {"hydroxyzine", "安泰乐"},
    # Z-drugs（镇静催眠）
    "唑吡坦": {"zolpidem", "思诺思"},
    "右佐匹克隆": {"eszopiclone", "艾司佐匹克隆"},
    "佐匹克隆": {"zopiclone", "忆孟返"},
    # ADHD 药物
    "哌甲酯": {"methylphenidate", "利他林", "专注达"},
    "托莫西汀": {"atomoxetine", "择思达"},
    "可乐定": {"clonidine"},
    "胍法辛": {"guanfacine", "intuniv"},
    # 辅助药物
    "苯海索": {"trihexyphenidyl", "安坦"},
    "普萘洛尔": {"propranolol", "心得安"},
    "苯巴比妥": {"phenobarbital", "鲁米那"},
    "扑米酮": {"primidone"},
    "加巴喷丁": {"gabapentin", "纽诺丁"},
    "普瑞巴林": {"pregabalin", "乐瑞卡"},
}

# 构建快速查找表
_DRUG_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canon, aliases in _PSYCH_DRUG_KEYWORDS.items():
    _DRUG_ALIAS_TO_CANONICAL[canon.lower()] = canon
    for alias in aliases:
        _DRUG_ALIAS_TO_CANONICAL[alias.lower()] = canon


class DrugNameMatcher:
    """精神科药物名称匹配器。

    维护两个集合：
    - known: 知识库已收录的药物（从 medication_knowledge.jsonl 自动提取）
    - detectable: 所有可识别的精神科药物（来自 _PSYCH_DRUG_KEYWORDS）
    """

    def __init__(self, jsonl_path: Optional[Path] = None):
        self._known: Set[str] = set()
        self._detectable: Set[str] = set(_PSYCH_DRUG_KEYWORDS.keys())
        self._load_known(jsonl_path)

    def _load_known(self, jsonl_path: Optional[Path] = None):
        """从 medication_knowledge.jsonl 提取已知药物名"""
        if jsonl_path is None:
            jsonl_path = (
                Path(__file__).resolve().parent.parent.parent
                / "data" / "knowledge" / "public" / "medication_knowledge.jsonl"
            )
        if not jsonl_path.exists():
            logger.warning("药物知识库文件不存在: %s", jsonl_path)
            return
        try:
            for line in jsonl_path.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                doc = json.loads(line)
                # 从 tags 中提取第一个标签（药物名）
                tags = doc.get("tags", [])
                if tags:
                    self._known.add(tags[0])
        except Exception:
            logger.warning("加载已知药物列表失败", exc_info=True)

    @property
    def known_drugs(self) -> Set[str]:
        return set(self._known)

    def detect_drugs(self, text: str) -> List[str]:
        """从用户查询中检测精神科药物名，返回规范化名称列表"""
        found = []
        text_lower = text.lower()
        # 先检查完整的中文名（多字词优先）
        for canon in sorted(_PSYCH_DRUG_KEYWORDS.keys(), key=lambda x: -len(x)):
            if canon in text:
                found.append(canon)
                continue
            # 检查别名
            for alias in _PSYCH_DRUG_KEYWORDS[canon]:
                if len(alias) < 2:
                    continue
                if alias.lower() in text_lower:
                    found.append(canon)
                    break
        # 去重保序
        seen = set()
        deduped = []
        for d in found:
            if d not in seen:
                seen.add(d)
                deduped.append(d)
        return deduped

    def find_unknown_drugs(self, text: str) -> List[str]:
        """返回查询中提到了但知识库未收录的药物名"""
        detected = self.detect_drugs(text)
        return [d for d in detected if d not in self._known]


# ---------------------------------------------------------------------------
# Tavily Search Provider
# ---------------------------------------------------------------------------

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilySearchProvider:
    """Tavily Search API 封装（REST，无三方 SDK 依赖）"""

    def __init__(self, api_key: Optional[str] = None):
        from config.settings import settings
        self._api_key = api_key or settings.TAVILY_API_KEY
        if not self._api_key or self._api_key.startswith("tvly-xxx"):
            logger.warning("TAVILY_API_KEY 未配置或使用占位符，搜索 fallback 将不可用")

    @property
    def available(self) -> bool:
        return bool(self._api_key) and not self._api_key.startswith("tvly-xxx")

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """执行搜索，返回 [{title, url, content}, ...]"""
        if not self.available:
            logger.warning("Tavily 不可用，跳过搜索")
            return []

        try:
            resp = requests.post(
                TAVILY_API_URL,
                json={
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": max_results,
                    "include_domains": ["dayi.org.cn"],
                    "include_answer": True,
                },
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            # 优先使用 answer（AI 摘要）
            if data.get("answer"):
                results.append({
                    "title": "概要",
                    "url": "",
                    "content": data["answer"],
                })
            # 追加搜索结果
            for r in data.get("results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                })
            return results
        except Exception:
            logger.warning("Tavily 搜索失败", exc_info=True)
            return []

    def search_medication(self, drug_name: str) -> str:
        """搜索药物信息并格式化为上下文片段"""
        query = f"{drug_name} 药品说明书 适应症 用法用量 不良反应 禁忌症 site:dayi.org.cn"
        results = self.search(query, max_results=3)

        if not results:
            return ""

        parts = [f"【{drug_name} - 实时搜索补充】\n以下信息由 Tavily Search API 实时检索，请交叉验证：\n"]
        for i, r in enumerate(results, 1):
            parts.append(f"来源 {i}: {r['title']}")
            if r['url']:
                parts.append(f"URL: {r['url']}")
            parts.append(f"内容: {r['content']}")
            parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# 核心判断逻辑
# ---------------------------------------------------------------------------

def search_unknown_drugs(
    query: str,
    retrieved_texts: List[str],
    drug_matcher: DrugNameMatcher,
    tavily: TavilySearchProvider,
) -> List[str]:
    """判断是否需要搜索 fallback 并返回搜索补充文本。

    Args:
        query: 用户原始查询
        retrieved_texts: 知识库检索到的文本列表
        drug_matcher: 药物名匹配器
        tavily: Tavily 搜索提供者

    Returns:
        搜索到的药物信息文本列表（空列表表示无需 fallback 或搜索无结果）
    """
    # 1. 检测查询中的未知药物
    unknown = drug_matcher.find_unknown_drugs(query)
    if not unknown:
        return []

    # 2. 逐个搜索未知药物
    supplements = []
    for drug in unknown:
        logger.info("知识库未覆盖药物，触发搜索 fallback: %s", drug)
        text = tavily.search_medication(drug)
        if text:
            supplements.append(text)
        time.sleep(0.5)  # 避免连续请求过密

    return supplements


def should_use_fallback(
    query: str,
    retrieved_texts: List[str],
    drug_matcher: DrugNameMatcher,
) -> bool:
    """快速判断是否需要搜索 fallback（不执行搜索）"""
    unknown = drug_matcher.find_unknown_drugs(query)
    return len(unknown) > 0
