"""周医生"情景→回应"风格参考检索器

从脱敏访谈语料构建的 zhou_style 索引中，按当前用户输入检索最相似的
"周医生当时怎么回应"样本，注入 DOCTOR_MODE prompt。

隐私边界：
- 索引由脱敏后的样本构建，不含真实患者 PII
- 注入 prompt 时附带"虚构化"声明，只学问法、不照搬个案内容

设计：
- 延迟加载（首次检索才打开 Chroma），索引缺失/加载失败 → 静默禁用，不阻断对话
- 失败返回空列表，调用方安全降级
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_COLLECTION = "zhou_style"
_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge" / "chroma_zhou_style"


class ZhouStyleRetriever:
    """周医生风格参考检索器（延迟加载，失败静默禁用）。"""

    def __init__(self, chroma_store=None):
        self._store = chroma_store  # 可注入（测试用）
        self._disabled = False

    # ── 内部：延迟加载 ──────────────────────────────────────

    def _ensure_store(self):
        if self._store is not None or self._disabled:
            return
        try:
            from config.settings import settings
            from core.rag.chroma_store import ChromaStore
            from core.rag.embedder import create_embedder

            embedder = create_embedder(backend=settings.EMBEDDING_BACKEND)
            self._store = ChromaStore(
                collection_name=_COLLECTION,
                embedding_fn=embedder,
                persist_dir=str(_PERSIST_DIR),
            )
            logger.info("ZhouStyle 索引已加载: %s (%d 条)",
                        _PERSIST_DIR, self._store._collection.count())  # noqa: SLF001
        except Exception:
            logger.warning("ZhouStyle 索引加载失败，本进程内禁用风格参考", exc_info=True)
            self._disabled = True

    # ── 公开 API ────────────────────────────────────────────

    def is_available(self) -> bool:
        self._ensure_store()
        return self._store is not None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """按用户输入检索最相似的风格样本。

        Returns:
            [{"human": ..., "doctor": ..., "score": ...}, ...] 按相似度降序
        """
        self._ensure_store()
        if self._store is None or not query.strip():
            return []

        try:
            results = self._store.search(query, top_k=top_k)
            out = []
            for r in results:
                text = r.get("text", "")
                if "医生：" not in text:
                    continue
                human, _, doctor = text.partition("医生：")
                human = human.replace("患者：", "", 1).strip()
                out.append({
                    "human": human,
                    "doctor": doctor.strip(),
                    "score": r.get("score", 0.0),
                })
            out.sort(key=lambda x: x["score"], reverse=True)
            return out
        except Exception:
            logger.warning("ZhouStyle 检索失败，返回空", exc_info=True)
            return []

    def format_for_prompt(self, hits: List[Dict], max_items: int = 2) -> str:
        """把命中的风格样本格式化为 prompt 注入段。

        只展示"周医生的接法"作为问法参考，并声明虚构化。
        """
        if not hits:
            return ""
        items = hits[:max_items]
        lines = [
            "## 参考：周医生遇到类似处境时的接话（已脱敏虚构化）",
            "以下是周医生以往接待类似处境时真实说过的话。**学他的问法和语气，",
            "不要照搬具体内容**（这些是脱敏虚构化的示例，仅参考风格）：",
        ]
        for i, h in enumerate(items, 1):
            human = h.get("human", "").strip()
            doctor = h.get("doctor", "").strip()
            if doctor:
                lines.append(f"{i}. 当来访者说「{human[:40]}」，周医生会回："
                             f"“{doctor[:120]}”")
        return "\n".join(lines)


# ── 单例 ────────────────────────────────────────────────────

_retriever: Optional[ZhouStyleRetriever] = None


def get_zhou_style_retriever() -> ZhouStyleRetriever:
    """获取 ZhouStyleRetriever 单例。"""
    global _retriever
    if _retriever is None:
        _retriever = ZhouStyleRetriever()
    return _retriever
