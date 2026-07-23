"""QueryRewriter 快速验证脚本 — 对比有/无改写的检索结果差异

用法（在 Docker 容器内）：
    docker compose -f docker/docker-compose.yml exec therapy-agent \
        python scripts/test_query_rewriter.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保项目根在 import 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.llm.base import get_llm_adapter
from core.rag.query_rewriter import QueryRewriter
from modules.intervention.rag.retriever import get_knowledge_retriever

# ---------------------------------------------------------------------------
# 10 条测试查询，覆盖不同知识类别
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    ("他最近老是不理我，是不是不爱了", "relationships"),
    ("一点小事就想发火，脾气越来越差", "coping_strategies"),
    ("躺床上三四个小时才能睡着", "sleep_health"),
    ("对什么都没兴趣，饭也不想吃", "disorder_knowledge"),
    ("我妈走了一年了还是天天想她", "grief_and_loss"),
    ("一到开会发言就手抖声音发抖", "psychology_basics"),
    ("吃氟西汀三个月胖了十斤正常吗", "medication_knowledge"),
    ("出车祸后不敢坐车了怎么办", "trauma_and_stress"),
    ("室友每天半夜打游戏我快疯了", "relationships"),
    ("总是觉得很累不想动", "disorder_knowledge"),
]


def print_doc(doc: str, max_len: int = 100) -> str:
    """截取文档前 max_len 字符，显示标题部分"""
    text = doc.replace("\n", " ").strip()
    return text[:max_len] + ("..." if len(text) > max_len else "")


def main():
    print("=" * 80)
    print("QueryRewriter 检索效果对比验证")
    print("=" * 80)

    # 1. 初始化
    print("\n[1/3] 加载知识库索引...")
    retriever = get_knowledge_retriever()

    print("[2/3] 创建 QueryRewriter...")
    llm_adapter = get_llm_adapter("openai_compatible")
    rewriter = QueryRewriter(llm_adapter.llm)

    # 2. 逐条对比
    print("[3/3] 开始对比测试\n")

    total_original = 0
    total_rewritten = 0

    for query, category in TEST_QUERIES:
        print(f"{'─' * 80}")
        print(f"查询: {query}")
        print(f"预期类别: {category}")

        # 无改写
        original_docs = retriever.retrieve(query, top_k=3)

        # 有改写 — 临时注入改写器
        retriever._rewriter = rewriter  # noqa: SLF001
        rewritten_docs = retriever.retrieve(query, top_k=3)
        retriever._rewriter = None  # 恢复  # noqa: SLF001

        print(f"\n  ┌─ 无改写 (top-3):")
        for i, doc in enumerate(original_docs, 1):
            print(f"  │ {i}. {print_doc(doc)}")
            total_original += 1
        if not original_docs:
            print("  │ (无结果)")

        print(f"  ├─ 有改写 (top-3):")
        for i, doc in enumerate(rewritten_docs, 1):
            print(f"  │ {i}. {print_doc(doc)}")
            total_rewritten += 1
        if not rewritten_docs:
            print("  │ (无结果)")

        # 差异统计
        orig_ids = {json.dumps(d[:80]) for d in original_docs}
        rewr_ids = {json.dumps(d[:80]) for d in rewritten_docs}
        same = orig_ids & rewr_ids
        new = rewr_ids - orig_ids
        removed = orig_ids - rewr_ids

        print(f"  └─ 相同: {len(same)} | 新增: {len(new)} | 消失: {len(removed)}")

    print(f"\n{'=' * 80}")
    print(f"总计: 无改写命中 {total_original} 条 | 有改写命中 {total_rewritten} 条")

    if total_rewritten > total_original:
        print(f"改写带来 +{total_rewritten - total_original} 条额外检索结果")
    elif total_rewritten == total_original:
        print("改写未改变检索结果数量")
    else:
        print("改写后结果减少（请检查改写质量）")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
