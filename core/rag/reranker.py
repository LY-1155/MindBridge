"""Reranker 模块：BGE 本地 + 百炼 qwen3-rerank API

用法：
    # 本地 BGE（需下载模型 ~1.5GB）
    reranker = BGEReranker(top_n=15, top_k=3)
    # 百炼 API（零本地资源）
    reranker = QwenReranker(top_n=15, top_k=3)
    # 通用接口
    top3 = reranker.rerank(query, docs)
"""

from __future__ import annotations

import json as _json
import logging
import os
from typing import List
from urllib import request as _request
from urllib.error import URLError as _URLError

logger = logging.getLogger(__name__)

# 国内 HuggingFace 镜像，加速模型下载
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 优先使用 ModelScope 本地缓存（国内镜像），回退 HF 在线下载
_MODELSCOPE_CACHE = "/app/models/models/BAAI--bge-reranker-v2-m3/snapshots/master"


class BGEReranker:
    """bge-reranker-v2-m3 Cross-Encoder 重排序器

    模型: BAAI/bge-reranker-v2-m3
      - 多语言（中英）cross-encoder
      - MIRACL / MTEB 多语言检索排行榜 SOTA 级
      - ~1.5GB 磁盘占用，GPU/CPU 均可运行
    """

    def __init__(self, top_n: int = 15, top_k: int = 3):
        self._top_n = top_n
        self._top_k = top_k
        self._model = None  # 延迟加载

    def _load(self):
        """延迟加载模型，避免启动时阻塞"""
        if self._model is not None:
            return
        logger.info("加载 bge-reranker-v2-m3 模型...")

        from FlagEmbedding import FlagReranker

        # 优先用 ModelScope 本地缓存，否则从 HF 在线下载
        if os.path.isdir(_MODELSCOPE_CACHE):
            model_path = _MODELSCOPE_CACHE
            logger.info("使用 ModelScope 缓存: %s", model_path)
        else:
            model_path = _MODEL_NAME
            logger.info("从 HuggingFace 下载: %s", _MODEL_NAME)

        self._model = FlagReranker(model_path, use_fp16=True)
        logger.info("bge-reranker-v2-m3 加载完成")

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """对文档列表重排序，返回 top_k 条。

        Args:
            query: 用户原始查询（非改写后的）
            docs: 粗检索返回的文档列表

        Returns:
            top_k 条按相关性降序排列的文档
        """
        if not docs:
            return []

        if len(docs) <= self._top_k:
            return docs

        self._load()

        candidates = docs[:self._top_n]

        # 构建 query-doc pairs
        pairs = [[query, doc[:800]] for doc in candidates]  # 截断过长文档

        try:
            scores = self._model.compute_score(pairs, normalize=True)
        except Exception:
            logger.warning("Reranker 模型推理失败，回退原始排序", exc_info=True)
            return candidates[:self._top_k]

        # 按分数降序
        scored = list(zip(scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored[:self._top_k]]


# ---------------------------------------------------------------------------
# 百炼 qwen3-rerank API（零本地资源，按量付费）
# ---------------------------------------------------------------------------

_QWEN_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/"
    "text-rerank/text-rerank"
)


class QwenReranker:
    """百炼 qwen3-rerank API 重排序器

    - 零本地模型，HTTP API 调用
    - 单次最多 500 文档，每篇最长 4000 tokens
    - 费用：约 ¥0.0035/次（15 文档标准调用）

    接口与 BGEReranker 完全一致，可直接替换。
    """

    def __init__(self, top_n: int = 15, top_k: int = 3):
        self._top_n = top_n
        self._top_k = top_k

    def rerank(self, query: str, docs: List[str]) -> List[str]:
        """对文档列表重排序，返回 top_k 条。"""
        if not docs:
            return []

        if len(docs) <= self._top_k:
            return docs

        candidates = docs[:self._top_n]

        from config.settings import settings

        body = _json.dumps({
            "model": settings.RERANK_MODEL_NAME,
            "input": {
                "query": query,
                "documents": candidates,
            },
            "parameters": {
                "top_n": self._top_k,
                "return_documents": True,
            },
        }).encode("utf-8")

        req = _request.Request(
            _QWEN_RERANK_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with _request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
        except _URLError:
            logger.warning("qwen3-rerank API 调用失败，回退原始排序", exc_info=True)
            return candidates[:self._top_k]

        if data.get("output") and data["output"].get("results"):
            # API 已按分数降序返回；document 是 {"text": "..."} 嵌套对象
            return [
                r["document"]["text"]
                for r in data["output"]["results"]
            ]
        else:
            logger.warning("qwen3-rerank 返回空结果，回退原始排序")
            return candidates[:self._top_k]
