"""知识库索引构建脚本：从 JSONL 数据建 Chroma 向量库

用法：
  python scripts/build_knowledge_index.py
  python scripts/build_knowledge_index.py --dry-run  # 只打印统计，不建索引

首次建索引时需确保 Ollama 已运行并拉取模型：
  ollama pull nomic-embed-text
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.rag.embedder import create_embedder
from core.rag.chroma_store import ChromaStore

CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "data" / "knowledge" / "chroma_index")
SOURCES_PATH = PROJECT_ROOT / "data" / "knowledge" / "sources.json"


def load_sources() -> List[dict]:
    if not SOURCES_PATH.exists():
        print(f"[错误] 配置文件不存在: {SOURCES_PATH}")
        return []

    config = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))
    return [layer for layer in config.get("layers", [])
            if layer.get("enabled") and layer.get("type") == "local"]


def _source_label(layer_id: str) -> str:
    """从 layer id 提取 source 标签"""
    if "private" in layer_id:
        return "private"
    if "public" in layer_id:
        return "public"
    return layer_id


def _category_from_filename(filename: str) -> str:
    """从 JSONL 文件名推导 category，如 clinical_knowledge.jsonl → clinical"""
    stem = Path(filename).stem
    # 去掉常见后缀
    for suffix in ["_knowledge", "_info", "_data"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def scan_jsonl(layer: dict) -> List[dict]:
    """扫描一个 layer 目录下的所有 JSONL，返回带 source/category 的条目"""
    data_dir = PROJECT_ROOT / layer["path"]
    source_label = _source_label(layer["id"])
    if not data_dir.exists():
        print(f"  [跳过] 目录不存在: {data_dir}")
        return []

    entries = []
    for f in sorted(data_dir.glob("*.jsonl")):
        category = _category_from_filename(f.name)
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc["_source_label"] = source_label
            doc["_category"] = category
            entries.append(doc)

    print(f"  {layer['name']}: {len(entries)} 条 (from {data_dir})")
    return entries


def build_chroma(entries: List[dict], backend: str = "api", dry_run: bool = False):
    """构建 Chroma 向量索引"""
    if not entries:
        return

    print(f"\n[建索引] 总共 {len(entries)} 条，目标: {CHROMA_PERSIST_DIR}")
    print(f"  Embedding 后端: {backend}")
    if dry_run:
        return

    embedder = create_embedder(backend=backend)

    store = ChromaStore(
        collection_name="knowledge_base",
        embedding_fn=embedder,
        persist_dir=CHROMA_PERSIST_DIR,
    )

    batch_size = 25
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        ids = [e.get("id", f"entry_{i + j}") for j, e in enumerate(batch)]
        texts = [
            f"{e.get('title', '')} | {' '.join(e.get('tags', []))} | {e.get('content', '')}"
            for e in batch
        ]
        metadatas = [
            {"source": e.get("_source_label", "public"),
             "category": e.get("_category", "general")}
            for e in batch
        ]

        store.upsert(ids=ids, texts=texts, metadatas=metadatas)
        pct = min(100, int((i + len(batch)) / len(entries) * 100))
        print(f"\r  进度: {pct}% ({min(i + len(batch), len(entries))}/{len(entries)})", end="", flush=True)
        time.sleep(0.1)

    print(f"\n  完成: {len(entries)} 条已写入 Chroma ({CHROMA_PERSIST_DIR})")


def main():
    parser = argparse.ArgumentParser(description="知识库索引构建")
    parser.add_argument("--dry-run", action="store_true", help="仅统计，不建索引")
    parser.add_argument("--backend", choices=["api", "local"], default="api",
                        help="Embedding 后端: api (远端 API) 或 local (本地 BGE-M3)")
    parser.add_argument("--api-base", help="百炼 API Base URL（默认 .env 中的）")
    parser.add_argument("--api-key", help="百炼 API Key（默认 .env 中的）")
    args = parser.parse_args()

    print("=== 知识库索引构建 ===\n")

    sources = load_sources()
    if not sources:
        print("[错误] 没有启用的知识源，请检查 sources.json 配置")
        return

    print(f"已启用的本地知识源: {len(sources)} 个\n")

    all_entries = []
    for src in sources:
        entries = scan_jsonl(src)
        if entries:
            all_entries.extend(entries)

    if not all_entries:
        print("\n[错误] 没有找到任何知识条目")
        return

    build_chroma(all_entries, backend=args.backend, dry_run=args.dry_run)

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
