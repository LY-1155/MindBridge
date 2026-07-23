"""QueryRewriter 量化验证 — 用 standard synthetic queries + ground truth 算 Recall@3/MRR

用法（Docker 容器内）：
    docker compose -f docker/docker-compose.yml exec therapy-agent \
        python scripts/eval_query_rewriter.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.llm.base import get_llm_adapter
from core.rag.query_rewriter import QueryRewriter
from modules.intervention.rag.retriever import get_knowledge_retriever

TEST_DATA = _PROJECT_ROOT / "data" / "eval" / "real_queries.jsonl"


# ── 指标 ──────────────────────────────────────────────────
def recall_at_k(retrieved_ids, relevant_ids, k=3):
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in relevant_ids if rid in set(retrieved_ids[:k]))
    return hits / len(relevant_ids)


def mrr(retrieved_ids, relevant_ids, k=None):
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in set(relevant_ids):
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids, relevant_ids, k=3):
    return 1.0 if any(rid in set(relevant_ids) for rid in retrieved_ids[:k]) else 0.0


# ── 文档 ID 提取 ───────────────────────────────────────────
def extract_doc_id(text: str, retriever) -> str | None:
    """从检索返回的文本反查文档 ID（通过 BM25 _docs 索引）"""
    bm25 = retriever._bm25  # noqa: SLF001
    for doc_id, doc_text in bm25._docs.items():  # noqa: SLF001
        if doc_text == text:
            return doc_id
    return None


# ── 主逻辑 ────────────────────────────────────────────────
def main():
    # 加载测试数据
    queries = []
    with open(TEST_DATA, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    # 抽样 15 条（全量 141 条太慢，抽样先验证趋势）
    import random
    random.seed(42)
    queries = random.sample(queries, min(15, len(queries)))

    print(f"抽样 {len(queries)} 条带 ground truth 的评测数据\n")

    # 初始化
    print("初始化检索器...")
    retriever = get_knowledge_retriever()

    print("创建 QueryRewriter (qwen-max)...")
    llm = get_llm_adapter("openai_compatible")
    rewriter = QueryRewriter(llm.llm)

    # 逐条评测
    results = {
        "without_rewrite": {"recall": [], "mrr": [], "hit_rate": []},
        "with_rewrite": {"recall": [], "mrr": [], "hit_rate": []},
    }
    by_category = defaultdict(lambda: {"without": [], "with": []})

    for i, item in enumerate(queries):
        query = item["query"]
        relevant = item.get("relevant_doc_ids", [])
        category = item.get("expected_categories", ["unknown"])[0]

        if not relevant:
            continue

        # 无改写
        docs_without = retriever.retrieve(query, top_k=3)
        ids_without = [extract_doc_id(d, retriever) for d in docs_without]
        ids_without = [x for x in ids_without if x]

        # 有改写
        retriever._rewriter = rewriter  # noqa: SLF001
        docs_with = retriever.retrieve(query, top_k=3)
        retriever._rewriter = None  # noqa: SLF001
        ids_with = [extract_doc_id(d, retriever) for d in docs_with]
        ids_with = [x for x in ids_with if x]

        # 算指标
        r_wo = recall_at_k(ids_without, relevant, 3)
        m_wo = mrr(ids_without, relevant)
        h_wo = hit_rate_at_k(ids_without, relevant, 3)

        r_wi = recall_at_k(ids_with, relevant, 3)
        m_wi = mrr(ids_with, relevant, 3)
        h_wi = hit_rate_at_k(ids_with, relevant, 3)

        results["without_rewrite"]["recall"].append(r_wo)
        results["without_rewrite"]["mrr"].append(m_wo)
        results["without_rewrite"]["hit_rate"].append(h_wo)
        results["with_rewrite"]["recall"].append(r_wi)
        results["with_rewrite"]["mrr"].append(m_wi)
        results["with_rewrite"]["hit_rate"].append(h_wi)

        by_category[category]["without"].append(r_wo)
        by_category[category]["with"].append(r_wi)

        # 每条打印
        delta_str = f"+{(r_wi - r_wo):+.2f}" if r_wi != r_wo else "  ="
        print(f"[{i+1:3d}/{len(queries)}] {delta_str} | {query[:50]}...")

    # ── 汇总 ──────────────────────────────────────────────
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    print()
    print("=" * 60)
    print("                  Recall@3    MRR      HitRate@3")
    print("-" * 60)
    wo = results["without_rewrite"]
    wi = results["with_rewrite"]
    print(f"  无改写          {avg(wo['recall']):.4f}     {avg(wo['mrr']):.4f}     {avg(wo['hit_rate']):.4f}")
    print(f"  有改写          {avg(wi['recall']):.4f}     {avg(wi['mrr']):.4f}     {avg(wi['hit_rate']):.4f}")
    print(f"  Delta           {avg(wi['recall'])-avg(wo['recall']):+.4f}     {avg(wi['mrr'])-avg(wo['mrr']):+.4f}     {avg(wi['hit_rate'])-avg(wo['hit_rate']):+.4f}")
    print("=" * 60)

    # 按类别
    print()
    print("按类别 Recall@3:")
    for cat in sorted(by_category.keys()):
        wo_cat = avg(by_category[cat]["without"])
        wi_cat = avg(by_category[cat]["with"])
        delta_cat = wi_cat - wo_cat
        print(f"  {cat:30s}  {wo_cat:.4f} → {wi_cat:.4f}  ({delta_cat:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
