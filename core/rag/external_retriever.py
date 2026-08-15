"""外部 API 检索器：从 sources.json external_api layer 加载配置。

当前策略：精神科药物名感知触发 → Tavily Search API 实时检索。
后续扩展：分数阈值通用 fallback（ADR 0008）。

集成点：KnowledgeRetriever.retrieve() — 检索收敛、截断到 top-k 后调用。
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from core.rag.search_fallback import (
    DrugNameMatcher,
    TavilySearchProvider,
    search_unknown_drugs,
)

logger = logging.getLogger(__name__)


class ExternalRetriever:
    """外部 API 检索器，封装搜索 fallback 逻辑。

    从 sources.json 的 external_api layer 配置加载。
    当前唯一 provider：Tavily Search API（药物名感知触发，覆盖 50+ 种精神科药物）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self._enabled = config.get("enabled", False)
        self._provider = config.get("provider", "")

        # 延迟创建，避免无网络或无 key 时浪费
        self._drug_matcher: Optional[DrugNameMatcher] = None
        self._tavily: Optional[TavilySearchProvider] = None

    @property
    def available(self) -> bool:
        if not self._enabled:
            return False
        if self._provider != "tavily":
            logger.warning("external_api provider 不支持: %s", self._provider)
            return False
        self._ensure_providers()
        return self._tavily is not None and self._tavily.available

    def _ensure_providers(self):
        """延迟初始化 provider（首次 search 或 available 检查时）"""
        if self._tavily is not None:
            return
        self._drug_matcher = DrugNameMatcher()
        self._tavily = TavilySearchProvider()

    def search(self, query: str, retrieved_texts: List[str]) -> List[str]:
        """执行外部搜索补充。

        Args:
            query: 用户原始查询
            retrieved_texts: 已有本地检索结果（用于判断是否已覆盖）

        Returns:
            搜索到的补充文本列表（空列表表示无需或搜索无结果）
        """
        if not self.available:
            return []

        try:
            return search_unknown_drugs(
                query=query,
                retrieved_texts=retrieved_texts,
                drug_matcher=self._drug_matcher,
                tavily=self._tavily,
            )
        except Exception:
            logger.warning("外部 API 搜索失败", exc_info=True)
            return []
