"""Embedding 模块：API 远端（百炼 DashScope） + 本地（BGE-M3）

向后兼容：QianwenEmbedding 保持原有行为不变。
新增 BGEM3Embedding 使用 BAAI/bge-m3 本地推理，零 API 费用。
"""

from __future__ import annotations

import logging
from typing import List
from chromadb import EmbeddingFunction

import openai

logger = logging.getLogger(__name__)

# 国内 HuggingFace 镜像
import os as _os
if "HF_ENDPOINT" not in _os.environ:
    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _resolve_model_path(name: str) -> str:
    """ModelScope 缓存优先 → HF 名称回退"""
    slug = name.replace("/", "--")
    candidate = f"/app/models/models/{slug}/snapshots/master"
    if _os.path.isdir(candidate):
        logger.info("使用 ModelScope 缓存: %s", candidate)
        return candidate
    return name


class QianwenEmbedding(EmbeddingFunction):
    """OpenAI 兼容 Embedding API 封装，实现 Chroma EmbeddingFunction 协议

    默认连接 Ollama 本地服务（nomic-embed-text）。
    也可通过 settings 切换为百炼 DashScope 等兼容 API。"""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        batch_size: int = 10,
        dimensions: int | None = None,
    ):
        from config.settings import settings
        self._client = openai.OpenAI(
            api_key=api_key or settings.EMBEDDING_API_KEY,
            base_url=api_base or settings.EMBEDDING_API_BASE,
        )
        self._model = model or settings.EMBEDDING_MODEL_NAME
        self._batch_size = batch_size
        self._dimensions = dimensions
        self._dim: int | None = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []

        all_vectors: list[list[float]] = []

        for i in range(0, len(input), self._batch_size):
            batch = input[i:i + self._batch_size]
            try:
                resp = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                    dimensions=self._dimensions,
                )
                batch_vecs = [d.embedding for d in resp.data]
            except Exception:
                logger.error("百炼 Embedding API 调用失败，回退到零向量", exc_info=True)
                dim = self._dim or 1024  # text-embedding-v3 / bge-m3 默认 1024 维
                batch_vecs = [[0.0] * dim for _ in batch]

            if not batch_vecs:
                continue

            if self._dim is None and batch_vecs[0]:
                self._dim = len(batch_vecs[0])

            all_vectors.extend(batch_vecs)

        return all_vectors


# ---------------------------------------------------------------------------
# BGE-M3 本地 Embedding（零 API 费用）
# ---------------------------------------------------------------------------

class BGEM3Embedding(EmbeddingFunction):
    """BGE-M3 本地稠密向量编码器，实现 Chroma EmbeddingFunction 协议

    - 模型: BAAI/bge-m3（多语言 1024-dim，8192 token 上下文）
    - 推理: FlagEmbedding 本地加载，FP16，零 API 费用
    - 首次加载约 5-15 秒（~4.2 GB 磁盘），之后常驻显存

    用法::

        embedder = BGEM3Embedding("BAAI/bge-m3")
        vectors = embedder(["文本1", "文本2"])  # → list[list[float]]
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 10,
    ):
        self._model_name = model_name
        self._batch_size = batch_size
        self._model = None
        self._dim = 1024  # bge-m3 dense embedding dimension

    def _load(self):
        if self._model is not None:
            return
        logger.info("加载 BGE-M3 Embedding 模型: %s ...", self._model_name)

        from FlagEmbedding import BGEM3FlagModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("使用设备: %s", device)

        model_path = _resolve_model_path(self._model_name)
        self._model = BGEM3FlagModel(
            model_path,
            use_fp16=(device == "cuda"),
            device=device,
        )
        logger.info("BGE-M3 Embedding 模型加载完成 (dim=%d)", self._dim)

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []

        self._load()

        all_vectors: list[list[float]] = []

        for i in range(0, len(input), self._batch_size):
            batch = input[i:i + self._batch_size]
            try:
                output = self._model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=8192,
                )
                batch_vecs = output["dense_vecs"].tolist()
            except Exception:
                logger.error("BGE-M3 Embedding 推理失败，回退到零向量", exc_info=True)
                batch_vecs = [[0.0] * self._dim for _ in batch]

            all_vectors.extend(batch_vecs)

        return all_vectors


def create_embedder(backend: str = "api") -> EmbeddingFunction:
    """工厂方法：根据配置创建 Embedding 实例。

    Args:
        backend: "api" → QianwenEmbedding (远端 API)
                 "local" → BGEM3Embedding (本地 bge-m3)

    Returns:
        ChromaDB EmbeddingFunction 实例
    """
    from config.settings import settings

    if backend == "local":
        return BGEM3Embedding(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            batch_size=10,
        )
    else:
        dim = settings.EMBEDDING_DIMENSIONS
        return QianwenEmbedding(
            dimensions=dim if dim > 0 else None,
        )
