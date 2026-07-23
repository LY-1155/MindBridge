"""RAG 检索层评估脚本。

6 组对照实验 × 3 个指标（Recall@3, MRR, Hit Rate@3）= 检索质量量化。

用法：
  # 完整评估（需要 LLM + embedding 服务运行中）
  python scripts/eval_retrieval.py --test-data data/eval/synthetic_queries.jsonl

  # 快速验证（假 LLM + 假 embedding，不需要 API）
  python scripts/eval_retrieval.py --test-data data/eval/synthetic_queries.jsonl --dry-run

  # LLM-as-judge Precision 验证
  python scripts/eval_retrieval.py --judge --judge-data data/eval/judge_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCES = _PROJECT_ROOT / "data" / "knowledge" / "sources.json"
_CHROMA_PERSIST_DIR = _PROJECT_ROOT / "data" / "knowledge" / "chroma_index"


# ==========================================================================
#  指标函数
# ==========================================================================

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 3) -> float:
    """Recall@k：前 k 个结果中命中了多少相关文档。

    |relevant ∩ top-k| / |relevant|
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = sum(1 for rid in relevant_ids if rid in top_k)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR (Mean Reciprocal Rank)：第一个相关文档的排名倒数。

    1 / rank_of_first_relevant  （0 if none found）
    """
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 3) -> float:
    """Hit Rate@k：至少命中 1 个相关文档则为 1，否则 0。"""
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0


# ==========================================================================
#  评估结果与运行器
# ==========================================================================

@dataclass
class EvalResult:
    """一组实验的聚合指标。"""
    name: str
    recall_at_3: float = 0.0
    mrr: float = 0.0
    hit_rate_at_3: float = 0.0
    num_queries: int = 0
    per_category: Dict[str, "EvalResult"] = field(default_factory=dict)

    @classmethod
    def from_scores(cls, name: str, recall_list: List[float],
                    mrr_list: List[float], hit_list: List[float]) -> "EvalResult":
        n = len(recall_list)
        if n == 0:
            return cls(name=name, num_queries=0)
        return cls(
            name=name,
            recall_at_3=sum(recall_list) / n,
            mrr=sum(mrr_list) / n,
            hit_rate_at_3=sum(hit_list) / n,
            num_queries=n,
        )


class RetrievalEvaluator:
    """检索评估器：构建检索器，运行 6 组实验，计算指标。"""

    def __init__(self, bm25, chroma, classifier, source_weights=None):
        self._bm25 = bm25
        self._chroma = chroma
        self._classifier = classifier
        self._hybrid = None  # lazy
        self._source_weights = source_weights or {}

    def _get_hybrid(self):
        if self._hybrid is None:
            from core.rag.hybrid_retriever import HybridRetriever
            self._hybrid = HybridRetriever(
                chroma_store=self._chroma,
                bm25_index=self._bm25,
                classifier=self._classifier,
                source_weights=self._source_weights,
                external_retriever=None,  # 评估时不触发外部 API
            )
        return self._hybrid

    def _run_group(self, test_cases: List[dict],
                   retrieve_fn: Callable[[str, int], List[dict]],
                   name: str) -> EvalResult:
        """对一组测试用例跑一次检索并计算指标。"""
        recall_scores = []
        mrr_scores = []
        hit_scores = []
        cat_recalls: Dict[str, List[float]] = defaultdict(list)
        cat_mrrs: Dict[str, List[float]] = defaultdict(list)
        cat_hits: Dict[str, List[float]] = defaultdict(list)

        for case in test_cases:
            query = case["query"]
            relevant_ids = case.get("relevant_doc_ids", [])
            if not relevant_ids:
                continue
            expected_cats = case.get("expected_categories", ["unknown"])

            try:
                docs = retrieve_fn(query, 3)
            except Exception:
                logger.warning("检索失败: %s", query[:50], exc_info=True)
                continue

            retrieved_ids = [d["id"] for d in docs]

            recall_scores.append(recall_at_k(retrieved_ids, relevant_ids))
            mrr_scores.append(mrr(retrieved_ids, relevant_ids))
            hit_scores.append(hit_rate_at_k(retrieved_ids, relevant_ids))

            for cat in expected_cats:
                cat_recalls[cat].append(recall_at_k(retrieved_ids, relevant_ids))
                cat_mrrs[cat].append(mrr(retrieved_ids, relevant_ids))
                cat_hits[cat].append(hit_rate_at_k(retrieved_ids, relevant_ids))

        result = EvalResult.from_scores(name, recall_scores, mrr_scores, hit_scores)

        for cat in cat_recalls:
            result.per_category[cat] = EvalResult.from_scores(
                f"{name}/{cat}",
                cat_recalls[cat],
                cat_mrrs[cat],
                cat_hits[cat],
            )

        return result

    def evaluate_all(self, test_cases: List[dict]) -> Dict[str, EvalResult]:
        """运行全部 6 组对照实验。"""
        hybrid = self._get_hybrid()
        from core.rag.query_classifier import QueryClassification

        results = {}

        # --- 组 1: dense_bare (Chroma-only, 无分类器) ---
        logger.info("组 1/6: dense_bare")
        results["dense_bare"] = self._run_group(
            test_cases, hybrid.retrieve_dense_only, "dense_bare")

        # --- 组 2: sparse_bare (BM25-only, 无分类器) ---
        logger.info("组 2/6: sparse_bare")
        results["sparse_bare"] = self._run_group(
            test_cases, hybrid.retrieve_sparse_only, "sparse_bare")

        # --- 组 3: dense_full (Chroma + classifier) ---
        logger.info("组 3/6: dense_full")
        results["dense_full"] = self._run_group(
            test_cases,
            lambda q, k: self._search_chroma_with_classifier(hybrid, q, k),
            "dense_full")

        # --- 组 4: sparse_full (BM25 + classifier) ---
        logger.info("组 4/6: sparse_full")
        results["sparse_full"] = self._run_group(
            test_cases,
            lambda q, k: self._search_bm25_with_classifier(hybrid, q, k),
            "sparse_full")

        # --- 组 5: hybrid_bare (RRF, 无分类器 --- 用 FakeLLM "all"/[]) ---
        logger.info("组 5/6: hybrid_bare")
        results["hybrid_bare"] = self._run_group(
            test_cases,
            lambda q, k: self._retrieve_hybrid_bare(hybrid, q, k),
            "hybrid_bare")

        # --- 组 6: hybrid_full (完整管线，含分类器) ---
        logger.info("组 6/6: hybrid_full")
        results["hybrid_full"] = self._run_group(
            test_cases,
            lambda q, k: self._retrieve_hybrid_full(hybrid, q, k),
            "hybrid_full")

        return results

    # -- 内置 helper：带分类器过滤的单路检索 --

    def _search_chroma_with_classifier(self, hybrid, query: str, top_k: int) -> List[dict]:
        classification = self._classifier.classify(query)
        filter_meta = hybrid._build_filter(classification)  # noqa: SLF001
        return hybrid._search_chroma(query, top_k=top_k, chroma_filter=filter_meta)  # noqa: SLF001

    def _search_bm25_with_classifier(self, hybrid, query: str, top_k: int) -> List[dict]:
        classification = self._classifier.classify(query)
        filter_meta = hybrid._build_filter(classification)  # noqa: SLF001
        return hybrid._search_bm25(query, top_k=top_k, bm25_filter=filter_meta)  # noqa: SLF001

    def _retrieve_hybrid_bare(self, hybrid, query: str, top_k: int) -> List[dict]:
        """模拟无分类器的 hybrid：分类器返回 "all"/[] = 不过滤。"""
        from core.rag.query_classifier import QueryClassification
        filter_meta = hybrid._build_filter(  # noqa: SLF001
            QueryClassification(source="all", categories=[]))
        dense = hybrid._search_chroma(query, top_k=top_k, chroma_filter=filter_meta)  # noqa: SLF001
        sparse = hybrid._search_bm25(query, top_k=top_k, bm25_filter=filter_meta)  # noqa: SLF001
        return hybrid._rrf_fuse(dense, sparse, top_k=top_k)  # noqa: SLF001

    def _retrieve_hybrid_full(self, hybrid, query: str, top_k: int) -> List[dict]:
        """含分类器的 hybrid：完整管线（用于消融对照）。"""
        classification = self._classifier.classify(query)
        filter_meta = hybrid._build_filter(classification)  # noqa: SLF001
        dense = hybrid._search_chroma(query, top_k=top_k * 3, chroma_filter=filter_meta)  # noqa: SLF001
        sparse = hybrid._search_bm25(query, top_k=top_k * 3, bm25_filter=filter_meta)  # noqa: SLF001
        return hybrid._rrf_fuse(dense, sparse, top_k=top_k)  # noqa: SLF001


# ==========================================================================
#  LLM-as-Judge Precision 验证
# ==========================================================================

JUDGE_SYSTEM_PROMPT = """你是一个心理健康知识检索评估器。
给定用户查询和一篇检索到的文档，评定文档对查询的相关性：

- 2: 高度相关 — 直接回答查询问题
- 1: 部分相关 — 有一定参考价值，但不直接回答
- 0: 不相关 — 内容与查询无关

仅输出数字（0, 1, 或 2），不要任何其他文字。"""


def _make_llm():
    from core.llm.base import get_llm_adapter
    return get_llm_adapter("qwen")


def _make_fake_llm():
    """假 LLM 用于 --dry-run。"""
    class FakeLLM:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content='{"source": "all", "categories": []}')
    return FakeLLM()


def judge_relevance(llm, query: str, doc_text: str) -> int:
    """LLM 评判文档相关性，返回 0/1/2。"""
    from langchain_core.messages import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Query: {query}\n\nDocument: {doc_text[:1500]}"),
    ]
    response = llm.invoke(messages)
    raw = response.content if hasattr(response, "content") else str(response)
    try:
        return max(0, min(2, int(raw.strip())))
    except ValueError:
        # 提取第一个数字
        import re
        m = re.search(r'\b([012])\b', raw)
        return int(m.group(1)) if m else 0


def run_judge_eval(args):
    """LLM-as-judge：手写查询 → 混合检索 → LLM 评判相关性 → Precision@3。"""
    judge_data_path = _PROJECT_ROOT / args.judge_data
    if not judge_data_path.exists():
        logger.error("Judge 数据文件不存在: %s", judge_data_path)
        sys.exit(1)

    test_cases = _load_test_data(judge_data_path)
    if not test_cases:
        logger.error("未加载到任何 judge 查询")
        sys.exit(1)

    llm = _make_fake_llm() if args.dry_run else _make_llm()
    bm25, chroma, classifier, source_weights = _build_retriever_components(dry_run=args.dry_run)

    from core.rag.hybrid_retriever import HybridRetriever
    hybrid = HybridRetriever(
        chroma_store=chroma,
        bm25_index=bm25,
        classifier=classifier,
        source_weights=source_weights,
        external_retriever=None,
    )

    precision_scores = []
    bin_precision = []

    for case in test_cases:
        query = case["query"]
        try:
            docs = hybrid.retrieve_with_ids(query, top_k=3)
        except Exception:
            logger.warning("检索失败: %s", query[:50], exc_info=True)
            continue

        rel_count = 0
        for doc in docs[:3]:
            score = judge_relevance(llm, query, doc["text"])
            if score >= 1:
                rel_count += 1

        precision_scores.append(rel_count / min(3, max(1, len(docs))))
        bin_precision.append(1.0 if rel_count > 0 else 0.0)

    if not precision_scores:
        logger.error("无有效评估结果")
        return

    n = len(precision_scores)
    avg_precision = sum(precision_scores) / n
    avg_bin_precision = sum(bin_precision) / n
    print(f"\n--- LLM-as-Judge 评估 ({n} queries) ---")
    print(f"  Precision@3 (graded):  {avg_precision:.3f}")
    print(f"  Precision@3 (binary):  {avg_bin_precision:.3f}  (at least 1 relevant)")


# ==========================================================================
#  数据加载与组件构建
# ==========================================================================

def _load_sources(path: Optional[Path] = None) -> dict:
    p = path or _DEFAULT_SOURCES
    if not p.exists():
        return {"layers": [], "retrieval": {"top_k": 3}}
    return json.loads(p.read_text(encoding="utf-8"))


def _category_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for suffix in ["_knowledge", "_info", "_data"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem


def _load_test_data(path: Path) -> List[dict]:
    if not path.exists():
        logger.error("测试数据文件不存在: %s", path)
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    logger.info("加载 %d 条测试数据", len(cases))
    return cases


def _build_bm25_from_jsonl(sources_path=None) -> "BM25Index":
    """从 JSONL 构建 BM25 索引。逻辑与 KnowledgeRetriever._build_bm25_from_jsonl 一致。"""
    from core.rag.bm25_index import BM25Index

    config = _load_sources(sources_path or _DEFAULT_SOURCES)
    bm25 = BM25Index()

    for layer in config.get("layers", []):
        if not layer.get("enabled") or layer.get("type") != "local":
            continue

        source_label = "private" if "private" in layer["id"] else "public"
        data_dir = Path(layer["path"])
        if not data_dir.is_absolute():
            data_dir = _PROJECT_ROOT / data_dir

        if not data_dir.exists():
            continue

        for f in sorted(data_dir.glob("*.jsonl")):
            category = _category_from_filename(f.name)
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = doc.get("id", f"{source_label}_{len(bm25._docs)}")  # noqa: SLF001
                text = f"{doc.get('title', '')} | {' '.join(doc.get('tags', []))} | {doc.get('content', '')}"
                bm25.add(doc_id, text, {"source": source_label, "category": category})

    logger.info("BM25 索引: %d 条文档", len(bm25._docs))  # noqa: SLF001
    return bm25


def _build_retriever_components(dry_run: bool = False):
    """构建评估用检索组件。

    Returns: (bm25, chroma, classifier, source_weights)
    """
    from core.rag.chroma_store import ChromaStore
    from core.rag.query_classifier import QueryClassifier

    # BM25
    bm25 = _build_bm25_from_jsonl(_DEFAULT_SOURCES)

    # Chroma
    chroma = None
    if dry_run:
        chroma = _build_dry_run_chroma(bm25)
    elif os.path.isdir(str(_CHROMA_PERSIST_DIR)):
        try:
            from core.rag.embedder import QianwenEmbedding
            chroma = ChromaStore(
                collection_name="knowledge_base",
                embedding_fn=QianwenEmbedding(),
                persist_dir=str(_CHROMA_PERSIST_DIR),
            )
            logger.info("Chroma 持久化索引加载成功")
        except Exception:
            logger.warning("Chroma 加载失败，将仅用 BM25", exc_info=True)

    if chroma is None:
        logger.info("Chroma 不可用，使用空 in-memory 集合")
        from chromadb import EmbeddingFunction
        class DummyEmbed(EmbeddingFunction):
            def __call__(self, input):
                return [[0.0] * 384 for _ in input]
        chroma = ChromaStore(
            collection_name="knowledge_base",
            embedding_fn=DummyEmbed(),
            persist_dir=None,
        )

    # Classifier（单例，供 evaluator 和 hybrid 共享）
    if dry_run:
        classifier = QueryClassifier(llm=_make_fake_llm())
    else:
        classifier = QueryClassifier(llm=_make_llm())

    # Source weights
    config = _load_sources(_DEFAULT_SOURCES)
    source_weights = config.get("source_weights", {})

    return bm25, chroma, classifier, source_weights


def _build_dry_run_chroma(bm25):
    """构建 dry-run 用的 in-memory Chroma（假 embedding）。"""
    from core.rag.chroma_store import ChromaStore
    from chromadb import EmbeddingFunction
    import math

    class TestEmbedding(EmbeddingFunction):
        VOCAB = (
            list("焦虑抑郁认知行为疗法障碍干预治疗正念冥想睡眠失眠药物控制放松自我家庭沟通情绪压力管理") +
            list("认知行为疗法焦虑抑郁失眠正念药物家庭沟通情绪压力CBTDBTACT")
        )

        def __call__(self, input: list[str]) -> list[list[float]]:
            dim = len(self.VOCAB)
            vectors = []
            for text in input:
                v = [0.0] * dim
                for i, ch in enumerate(self.VOCAB):
                    if ch in text:
                        v[i] = float(text.count(ch)) / max(len(text), 1)
                norm = math.sqrt(sum(x * x for x in v))
                if norm > 0:
                    v = [x / norm for x in v]
                vectors.append(v)
            return vectors

    emb_fn = TestEmbedding()
    chroma = ChromaStore(collection_name="eval_test", embedding_fn=emb_fn, persist_dir=None)

    # 将 BM25 中的文档同步到 Chroma（dry-run 用）
    ids = list(bm25._docs.keys())  # noqa: SLF001
    texts = list(bm25._docs.values())  # noqa: SLF001
    metas = [bm25._meta.get(did, {}) for did in ids]  # noqa: SLF001
    if ids:
        chroma.add(ids=ids, texts=texts, metadatas=metas)
    logger.info("Dry-run Chroma: %d docs (假 embedding)", len(ids))
    return chroma


# ==========================================================================
#  输出与主函数
# ==========================================================================

def _print_report(results: Dict[str, EvalResult], categories: List[str] = None):
    """打印评估对比表。"""
    n = list(results.values())[0].num_queries if results else 0

    print(f"\n{'='*75}")
    print(f"  RAG 检索评估结果 ({n} queries, top_k=3)")
    print(f"{'='*75}")
    print(f"{'Group':<16} {'Recall@3':>9} {'MRR':>9} {'HitRate@3':>9}")
    print(f"{'-'*16} {'-'*9} {'-'*9} {'-'*9}")

    group_order = ["dense_bare", "sparse_bare", "dense_full", "sparse_full",
                   "hybrid_bare", "hybrid_full"]

    for name in group_order:
        if name not in results:
            continue
        r = results[name]
        print(f"{name:<16} {r.recall_at_3:>9.3f} {r.mrr:>9.3f} {r.hit_rate_at_3:>9.3f}")

    # Delta: full vs bare hybrid
    if "hybrid_full" in results and "hybrid_bare" in results:
        f = results["hybrid_full"]
        b = results["hybrid_bare"]
        print(f"{'-'*16} {'-'*9} {'-'*9} {'-'*9}")
        print(f"{'Delta (full-bare)':<16} "
              f"{f.recall_at_3 - b.recall_at_3:>+9.3f} "
              f"{f.mrr - b.mrr:>+9.3f} "
              f"{f.hit_rate_at_3 - b.hit_rate_at_3:>+9.3f}")

    # 消融分析
    print(f"\n消融分析：")
    if "hybrid_bare" in results and "dense_bare" in results:
        hb = results["hybrid_bare"]
        db = results["dense_bare"]
        sb = results.get("sparse_bare")
        print(f"  Hybrid vs Dense-only:       Recall {hb.recall_at_3 - db.recall_at_3:+.3f}")
        if sb:
            print(f"  Hybrid vs Sparse-only:      Recall {hb.recall_at_3 - sb.recall_at_3:+.3f}")
        if "dense_full" in results:
            df = results["dense_full"]
            print(f"  Classifier gain on Dense:   Recall {df.recall_at_3 - db.recall_at_3:+.3f}")
        if "sparse_full" in results:
            sf = results["sparse_full"]
            print(f"  Classifier gain on Sparse:  Recall {sf.recall_at_3 - sb.recall_at_3:+.3f}")

    # 按 category 分组
    if categories:
        print(f"\n按知识类别分组 (Recall@3):")
        print(f"{'Category':<28} {'dense_bare':>10} {'sparse_bare':>11} {'hybrid_full':>11}")
        print(f"{'-'*28} {'-'*10} {'-'*11} {'-'*11}")
        for cat in categories:
            d_recall = results.get("dense_bare").per_category.get(cat)
            s_recall = results.get("sparse_bare").per_category.get(cat)
            h_recall = results.get("hybrid_full").per_category.get(cat)
            d_val = f"{d_recall.recall_at_3:.3f}" if d_recall else "N/A"
            s_val = f"{s_recall.recall_at_3:.3f}" if s_recall else "N/A"
            h_val = f"{h_recall.recall_at_3:.3f}" if h_recall else "N/A"
            print(f"{cat:<28} {d_val:>10} {s_val:>11} {h_val:>11}")

    print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG 检索层评估")
    parser.add_argument("--test-data", type=str,
                        default="data/eval/synthetic_queries.jsonl",
                        help="测试数据路径（相对于项目根目录）")
    parser.add_argument("--dry-run", action="store_true",
                        help="使用假 LLM + 假 embedding，无需 API")
    parser.add_argument("--judge", action="store_true",
                        help="运行 LLM-as-judge Precision 验证")
    parser.add_argument("--judge-data", type=str,
                        default="data/eval/judge_queries.jsonl",
                        help="Judge 查询文件路径")
    parser.add_argument("--sample", type=int, default=0,
                        help="只跑前 N 条查询（0=全部）")
    args = parser.parse_args()

    # Judge 模式
    if args.judge:
        run_judge_eval(args)
        return

    # 加载测试数据
    test_data_path = _PROJECT_ROOT / args.test_data
    test_cases = _load_test_data(test_data_path)
    if not test_cases:
        logger.error("未加载到任何测试数据。请先生成：python scripts/generate_rag_test_data.py --dry-run")
        sys.exit(1)

    if args.sample > 0 and args.sample < len(test_cases):
        test_cases = test_cases[:args.sample]
        logger.info("抽样 %d 条查询用于评估", args.sample)

    # 构建检索组件
    bm25, chroma, classifier, source_weights = _build_retriever_components(dry_run=args.dry_run)

    evaluator = RetrievalEvaluator(
        bm25=bm25,
        chroma=chroma,
        classifier=classifier,
        source_weights=source_weights,
    )

    # 跑 6 组实验
    results = evaluator.evaluate_all(test_cases)

    # 收集所有 category
    all_cats = set()
    for r in results.values():
        all_cats.update(r.per_category.keys())

    _print_report(results, categories=sorted(all_cats) if all_cats else None)


if __name__ == "__main__":
    main()
