"""清理 zhou_style 索引中仍含可脱敏 PII 的样本

用于脱敏规则升级后清理旧索引（避免全量重建）。
扫描每条样本文本，若新脱敏规则仍能进一步脱敏 → 该样本含 PII → 删除。

用法：
  python scripts/purge_zhou_style_pii.py           # dry-run，只报告
  python scripts/purge_zhou_style_pii.py --execute # 实际删除
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from core.privacy.desensitize import desensitize
from core.rag.chroma_store import ChromaStore
from core.rag.embedder import create_embedder
from modules.intervention.rag.zhou_style import _COLLECTION, _PERSIST_DIR

BATCH = 1000


def collect_pii_ids() -> list[str]:
    store = ChromaStore(
        collection_name=_COLLECTION,
        embedding_fn=create_embedder(backend=settings.EMBEDDING_BACKEND),
        persist_dir=str(_PERSIST_DIR),
    )
    col = store._collection  # noqa: SLF001
    total = col.count()
    bad_ids: list[str] = []
    offset = 0
    while offset < total:
        got = col.get(limit=BATCH, offset=offset, include=["documents"])
        docs = got.get("documents") or []
        ids = got.get("ids") or []
        for doc_id, text in zip(ids, docs):
            if desensitize(text) != text:
                bad_ids.append(doc_id)
        offset += len(docs)
        if len(docs) == 0:
            break
    return bad_ids


def main():
    parser = argparse.ArgumentParser(description="清理 zhou_style 索引中残留 PII 样本")
    parser.add_argument("--execute", action="store_true",
                        help="实际删除（默认 dry-run）")
    args = parser.parse_args()

    store = ChromaStore(
        collection_name=_COLLECTION,
        embedding_fn=create_embedder(backend=settings.EMBEDDING_BACKEND),
        persist_dir=str(_PERSIST_DIR),
    )
    col = store._collection  # noqa: SLF001
    total = col.count()

    print(f"索引总数: {total}")
    bad = collect_pii_ids()
    print(f"含 PII 样本: {len(bad)} ({len(bad) / max(total, 1) * 100:.1f}%)")

    if not bad:
        print("✅ 索引已干净，无需清理")
        return

    if not args.execute:
        print("(dry-run) 如需删除请加 --execute")
        return

    col.delete(ids=bad)
    new_total = col.count()
    print(f"已删除 {len(bad)} 条，剩余 {new_total} 条")


if __name__ == "__main__":
    main()
