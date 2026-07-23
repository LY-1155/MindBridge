"""RAG 全链路测试脚本：QR → Union(Chroma+BM25) → Reranker → top-K

用法:
    python scripts/test_rag_full.py "分手后一直走不出来怎么办"
    python scripts/test_rag_full.py "分手后一直走不出来怎么办" -k 5
    python scripts/test_rag_full.py "分手后一直走不出来怎么办" -k 3 5 7 10
    python scripts/test_rag_full.py                        # 交互式输入
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.intervention.rag.retriever import get_knowledge_retriever
from core.llm.base import get_llm_adapter
from core.rag.query_rewriter import QueryRewriter
from core.rag.reranker import QwenReranker


def test_pipeline(query: str, k_values: list[int] = None):
    if k_values is None:
        k_values = [3]

    # ── 初始化 ──
    print("初始化...", end=" ", flush=True)
    retriever = get_knowledge_retriever()
    retriever._ensure_hybrid()
    llm = get_llm_adapter("openai_compatible")
    retriever._rewriter = QueryRewriter(llm.llm)
    print("OK")

    print(f"\n{'='*60}")
    print(f"查询: {query}")
    print(f"{'='*60}")

    # Query Rewriting
    rewritten = retriever._rewriter.rewrite(query)
    print(f"\n[QR] {rewritten[:150]}...")

    # 不同 K 值对比
    for k in k_values:
        retriever._reranker = QwenReranker(top_n=20, top_k=k)
        results = retriever.retrieve(query, top_k=k)
        print(f"\n── K={k} ({len(results)} 篇) ──")
        for i, doc in enumerate(results, 1):
            parts = doc.split(" | ", 2)
            title = parts[0] if parts else ""
            content = parts[2][:100] if len(parts) > 2 else doc[:100]
            print(f"  [{i}] {title}")
            print(f"      {content}...")


if __name__ == "__main__":
    args = sys.argv[1:]

    # 解析 -k 参数
    k_values = [3]
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "-k":
            k_values = []
            i += 1
            while i < len(args) and args[i].isdigit():
                k_values.append(int(args[i]))
                i += 1
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts).strip()
    if not query:
        query = input("请输入问题: ").strip()
    if not query:
        query = "分手后一直走不出来怎么办"

    test_pipeline(query, k_values)
