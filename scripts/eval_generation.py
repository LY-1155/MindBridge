"""RAG 生成侧评估（RAGAS 0.4.3）——周医生真实门诊语料测试集

评测闭环：检索（QR→BM25 / 生产管线）→ 生成（grounded-QA + 周医生 persona）→ RAGAS 双 judge 打分。

指标：
  - Faithfulness      忠实度（生成回答 vs 检索资料）       —— judge LLM
  - ContextRelevance  上下文相关性（检索资料 vs 用户问题） —— judge LLM
  - SemanticSimilarity 医生金标准相似度（persona 回答 vs 真实医生回应）—— bge-m3 embedding，无 judge

双 judge 交叉验证（防"AI 自己给自己打分"）：
  - Run A（独立 judge）: qwen3.7-flash   —— 简历数字取此
  - Run B（自评 judge）: qwen3.7-max     —— 与生成同模型，量化分歧

用法：
  python scripts/eval_generation.py --build-test-set -n 40            # 只构建测试集
  python scripts/eval_generation.py --smoke                            # 5 条冒烟
  python scripts/eval_generation.py -n 40 --retriever fast             # 全量（默认）
  python scripts/eval_generation.py -n 40 --retriever production       # 全量（BGE 重排）

隐私：测试集只写脱敏后的患者发言 + 医生回应，与 chroma_zhou_style 索引同标准。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import types as _types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── ragas 0.4.3 打包 bug：langchain_community.chat_models.vertexai 未声明依赖 ──
# 必须在 import ragas 之前注册 shim（只用 OpenAI 兼容接口，ChatVertexAI 永不实例化）。
_vertexai_mod = _types.ModuleType("langchain_community.chat_models.vertexai")
class _ChatVertexAI:  # noqa: N801
    pass
_vertexai_mod.ChatVertexAI = _ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_mod

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

INTERVIEW_ROOT = PROJECT_ROOT / "心理医生访谈数据"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
TEST_SET_PATH = EVAL_DIR / "zhou_queries.jsonl"
RESULTS_JSON = PROJECT_ROOT / "scripts" / "eval_generation_results.json"
RESULTS_MD = PROJECT_ROOT / "scripts" / "eval_generation_results.md"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_generation")

from config.settings import settings
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.metrics.collections import Faithfulness, ContextRelevance, SemanticSimilarity

from core.rag.embedder import QianwenEmbedding
from core.llm.base import OpenAICompatibleAdapter, LLMConfig
from core.privacy.desensitize import desensitize_turn
from build_zhou_style_index import extract_turns, is_noise

# ════════════════════════════════════════════════════════════════════
#  1. 测试集构建（周医生语料 → 脱敏 → LLM 精选自包含陈述 → 分层抽样）
# ════════════════════════════════════════════════════════════════════
# 真实对话轮次碎片化、依赖上下文，直接抽会得到"对，在一个房间的时候……"这种无法
# 独立成查询的句子。流程：启发式收窄池 → qwen3.7-flash 批量判断"是否自包含的心理
# 困扰陈述"（1/0）→ 只保留 1 → 按 source_file 分层抽样。LLM 判定结果缓存，可复现。

_MIN_QUERY_LEN = 25      # 患者发言脱敏后（去标点）最短长度
_MIN_REF_LEN = 20        # 医生回应最短长度（金标准要有信息量）
_DOCTOR_SKIP_MARKERS = ("```", "http://", "https://", "治疗建议：", "干预方案：",
                        "1.", "2.", "3.", "①", "②", "③")
_META_MARKERS = ("发言人", "[患者]", "你听明白", "明白吗", "你说呢", "你懂",
                 "对不对", "对吧", "是吗", "好不好", "我跟你说", "我说我说",
                 "他问我", "你问我", "你刚才说", "您说", "妈妈你听明白")
# 出现任一即认为有心理困扰主题（用于启发式收窄，最终由 LLM 判定）
_PROBLEM_KW = ("焦虑", "紧张", "睡", "梦", "哭", "难过", "烦", "累", "压力",
               "害怕", "担心", "抑郁", "情绪", "没意思", "不想", "控制不住",
               "头疼", "肚子", "夫妻", "吵架", "离婚", "孩子", "儿子", "女儿",
               "妈妈", "爸爸", "父母", "婆媳", "同学", "朋友", "同事", "成绩",
               "考试", "学习", "工作", "辞职", "对象", "男朋友", "女朋友",
               "分手", "结婚", "孤单", "社交", "暴食", "食欲", "自残", "割腕",
               "自杀", "想死", "活着", "希望", "发脾气", "摔东西", "打人",
               "不上学", "休学", "拒学", "沉迷", "游戏", "手机", "注意力", "发呆")
_SEED = 20260811
_CANDIDATES_PATH = EVAL_DIR / "zhou_candidates.jsonl"   # 启发式池 + LLM 判定（缓存）
_CURATE_BATCH = 25


def _clean_len(text: str) -> int:
    return len(re.sub(r"[\s，。、,.！？!?；;：:·…——“”‘’（）()\[\]{}　]+", "", text))


def _build_candidates() -> List[Dict[str, str]]:
    """启发式收窄候选池：脱敏 + 过滤 + 去重，带 source_file。"""
    pairs: List[Dict[str, str]] = []
    seen: set[str] = set()
    for p in sorted(Path(INTERVIEW_ROOT).rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for human_raw, doctor_raw in extract_turns(data):
            if _clean_len(human_raw) < _MIN_QUERY_LEN or _clean_len(doctor_raw) < _MIN_REF_LEN:
                continue
            if is_noise(human_raw, _MIN_QUERY_LEN) or is_noise(doctor_raw, _MIN_REF_LEN):
                continue
            if any(m in doctor_raw for m in _DOCTOR_SKIP_MARKERS):
                continue
            human = desensitize_turn("human", human_raw)
            doctor = desensitize_turn("assistant", doctor_raw)
            if not human or not doctor or _clean_len(human) < _MIN_QUERY_LEN:
                continue
            if any(m in human for m in _META_MARKERS):
                continue
            if not any(k in human for k in _PROBLEM_KW) and "？" not in human and "?" not in human:
                continue
            key = hashlib.sha1(human.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "source_file": p.name,
                "query": human,
                "reference_answer": doctor,
            })
    return pairs


def _curator_llm():
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: F401
    # 精选用主模型（qwen3.7-max）：判准明显高于 flash，一次性离线成本可接受
    return OpenAICompatibleAdapter(config=LLMConfig(
        model_name=settings.MODEL_NAME, temperature=0, max_tokens=1024,
        streaming=False, timeout=120,
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )).llm


_CURATE_SYSTEM = (
    "想象这句话是一个陌生人在网上心理咨询服务里发出的第一条消息。"
    "判断咨询师仅凭这一句，能否明白这个人正在为什么事或什么情绪困扰，并给出有帮助的回应。\n"
    "判 0：像是聊天中途的接话（对/嗯/但/所以/那 开头）；指代不明（他/她/这个 没交代是谁）；"
    "语音转写乱码；只是讲了件发生的事却看不出困扰主题（纯事件叙述）；自言自语式碎句。\n"
    "判 1：像真实求助的开场，单独读就能看出困扰主题（亲子、睡眠、情绪、学业、工作、社交、"
    "夫妻、健康等）。\n"
    "输出：每行一个，1:主题 或 0，严格按顺序，不要输出任何其他文字。"
)


def _curate_batch(llm, batch: List[Dict[str, str]]) -> tuple[List[int], Dict[int, str]]:
    """对一批候选调用 LLM 判定自包含性，返回 (0/1 列表, {索引: 主题})。"""
    body = "\n".join(f"[{i+1}] {c['query']}" for i, c in enumerate(batch))
    for attempt in range(2):
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            resp = llm.invoke([SystemMessage(content=_CURATE_SYSTEM),
                               HumanMessage(content=body)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            flags: List[int] = []
            themes: Dict[int, str] = {}
            # 格式 1：JSON 数组 [1,0,...]
            m = re.search(r"\[.*\]", text, re.S)
            if m:
                arr = json.loads(m.group(0))
                flags = [1 if int(v) == 1 else 0 for v in arr]
            else:
                # 格式 2：每行一个 "0" 或 "1:主题"（qwen3.7-max 实测偏好）
                for line in text.splitlines():
                    mm = re.match(r"\s*([01])(?::(.*))?\s*$", line)
                    if mm:
                        flags.append(1 if mm.group(1) == "1" else 0)
                        if mm.group(1) == "1" and mm.group(2):
                            themes[len(flags) - 1] = mm.group(2).strip()
            if not flags:
                raise ValueError("empty flags")
            if len(flags) != len(batch):
                logger.warning("curate 数量不匹配：模型 %d 行 / 批 %d 条，补齐处理",
                               len(flags), len(batch))
            # 数量不足时补 1（保底保留），多余截断
            flags = (flags + [1] * len(batch))[:len(batch)]
            return flags, themes
        except Exception as e:
            logger.warning("curate batch 解析失败（第 %d 次）: %s", attempt + 1, e)
    return [1] * len(batch), {}  # 两次失败保底全保留


def _load_or_curate_candidates(force: bool = False) -> List[Dict[str, str]]:
    """加载缓存候选；无缓存或 force 时构建 + LLM 精选并写缓存。"""
    if _CANDIDATES_PATH.exists() and not force:
        rows = [json.loads(line) for line in _CANDIDATES_PATH.read_text(encoding="utf-8").splitlines() if line]
        logger.info("复用候选池 %s（%d 条）", _CANDIDATES_PATH, len(rows))
        return rows
    pool = _build_candidates()
    logger.info("启发式候选池：%d 条，开始 LLM 精选…", len(pool))
    llm = _curator_llm()
    t0 = time.time()
    for i in range(0, len(pool), _CURATE_BATCH):
        batch = pool[i:i + _CURATE_BATCH]
        flags, themes = _curate_batch(llm, batch)
        for j, (c, flag) in enumerate(zip(batch, flags)):
            c["self_contained"] = flag
            if flag and j in themes:
                c["theme"] = themes[j]
        pct = min(100, int((i + len(batch)) / len(pool) * 100))
        if (i // _CURATE_BATCH) % 10 == 0:
            print(f"  精选进度: {pct}% ({min(i + len(batch), len(pool))}/{len(pool)})", flush=True)
    n_pass = sum(1 for c in pool if c.get("self_contained") == 1)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CANDIDATES_PATH, "w", encoding="utf-8") as f:
        for c in pool:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("LLM 精选完成：%d 条 → 自包含 %d 条（%.0f%%），耗时 %.0fs，缓存 %s",
                len(pool), n_pass, 100 * n_pass / len(pool), time.time() - t0, _CANDIDATES_PATH)
    return pool


def _stratified_sample(rows: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    """按 source_file 分层轮转抽样，避免单一会诊主导。"""
    if len(rows) <= n:
        return rows
    rng = random.Random(_SEED)
    by_file: Dict[str, List[Dict[str, str]]] = {}
    for p in rows:
        by_file.setdefault(p["source_file"], []).append(p)
    for key in by_file:
        rng.shuffle(by_file[key])
    picked: List[Dict[str, str]] = []
    i = 0
    while len(picked) < n:
        changed = False
        for key in list(by_file.keys()):
            if len(picked) >= n:
                break
            if i < len(by_file[key]):
                picked.append(by_file[key][i])
                changed = True
        if not changed:
            break
        i += 1
    return picked[:n]


def build_test_set(n: int, force: bool = False) -> List[Dict[str, str]]:
    """构建/加载测试集（LLM 精选自包含陈述 → 分层抽样）。force=True 重建候选与抽样。"""
    if TEST_SET_PATH.exists() and not force:
        rows = [json.loads(line) for line in TEST_SET_PATH.read_text(encoding="utf-8").splitlines() if line]
        logger.info("复用已有测试集 %s（%d 条）", TEST_SET_PATH, len(rows))
        return rows
    candidates = _load_or_curate_candidates(force=force)
    # 只取"真判 1"（有 theme）的候选：无 theme 的是批次解析失败时保底全留的污染项，
    # 不能进测试集（见 2026-08-11 精选 403 配额耗尽事件）
    passed = [c for c in candidates if c.get("self_contained") == 1 and c.get("theme")]
    if not passed:
        logger.warning("LLM 精选后没有可用候选，回退到启发式池直接抽样")
        passed = candidates
    sample = _stratified_sample(passed, n)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_SET_PATH, "w", encoding="utf-8") as f:
        for i, s in enumerate(sample):
            f.write(json.dumps({
                "id": f"zhou_{i:03d}",
                "query": s["query"],
                "reference_answer": s["reference_answer"],
                "source_file": s["source_file"],
                "theme": s.get("theme", ""),
            }, ensure_ascii=False) + "\n")
    logger.info("测试集已写入 %s（%d 条，候选 %d → 自包含 %d）",
                TEST_SET_PATH, len(sample), len(candidates), len(passed))
    return sample


# ════════════════════════════════════════════════════════════════════
#  2. LLM / Embedding 客户端
# ════════════════════════════════════════════════════════════════════

def _make_langchain(model: str, temperature: float, max_tokens: int,
                    thinking_off: bool = True):
    """LangChain ChatOpenAI（生成/改写用）。"""
    return OpenAICompatibleAdapter(config=LLMConfig(
        model_name=model, temperature=temperature, max_tokens=max_tokens,
        streaming=False, timeout=90,
        model_kwargs={"extra_body": {"enable_thinking": False}} if thinking_off else None,
    )).llm


def _make_ragas_judge(model: str):
    """ragas InstructorLLM judge（qwen3.x 关思考提速）。

    judge 走独立配置 JUDGE_MODEL_NAME/JUDGE_API_KEY/JUDGE_API_BASE（deepseek-v4-flash）；
    未配置时回退到 OPENAI_*（旧 qwen judge 行为）。
    """
    api_key = settings.JUDGE_API_KEY or settings.OPENAI_API_KEY
    api_base = settings.JUDGE_API_BASE or settings.OPENAI_API_BASE
    judge_model = settings.JUDGE_MODEL_NAME or model
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)
    return llm_factory(
        judge_model,
        provider="openai",
        client=client,
        extra_body={"enable_thinking": False},
        max_tokens=4096,  # 默认过低：Faithfulness 抽取 claims 时触发 max_tokens 截断（smoke 实测 zhou_002 失败）；2048 仍会触顶（2026-08-16 production smoke 5 条中 3 条双 judge 全失），提到 4096
    )


class ZhouEmbedder(BaseRagasEmbedding):
    """把项目 QianwenEmbedding（callable list[str]->list[list[float]]）包成 ragas BaseRagasEmbedding。"""

    def __init__(self, embed_fn):
        super().__init__()
        self.embed_fn = embed_fn

    def embed_text(self, text: str, **kwargs) -> List[float]:
        return self.embed_fn([text])[0]

    async def aembed_text(self, text: str, **kwargs) -> List[float]:
        return self.embed_fn([text])[0]


# ════════════════════════════════════════════════════════════════════
#  3. 检索（两种模式）
# ════════════════════════════════════════════════════════════════════

class FastRetriever:
    """fast：QueryRewrite → BM25 top-3（RAG 优化记录实测最优，秒级）。"""

    def __init__(self, rewriter):
        from modules.intervention.rag.retriever import _build_bm25_from_jsonl
        self._bm25 = _build_bm25_from_jsonl()
        self._rewriter = rewriter

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        search_query = self._rewriter.rewrite(query)
        raw = self._bm25.search(search_query, top_k=top_k)
        texts = [self._bm25._docs.get(did, "") for did, _ in raw]  # noqa: SLF001
        return [t for t in texts if t][:top_k]


class ProductionRetriever:
    """production：QR → Chroma+BM25 并集 → BGE 重排 top-3（与生产一致，慢）。"""

    def __init__(self, rewriter):
        from core.rag.reranker import BGEReranker
        from modules.intervention.rag.retriever import KnowledgeRetriever
        self._retriever = KnowledgeRetriever(
            rewriter=rewriter,
            reranker=BGEReranker(top_n=15, top_k=3),
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        return self._retriever.retrieve(query, top_k=top_k)


# ════════════════════════════════════════════════════════════════════
#  4. 生成（两种模式）
# ════════════════════════════════════════════════════════════════════

_GROUNDED_SYSTEM = """你是一个严格基于检索资料的问答助手。请根据【检索资料】用中文回答【用户问题】。

规则：
1. 只使用检索资料中的信息，不编造资料中没有的内容
2. 资料不足以回答时，明确回答"资料中没有相关信息"
3. 回答简洁（2-4 句话），直接给出答案"""


def _generate_grounded(llm, query: str, contexts: List[str]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    ctx = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts)) if contexts else "（无检索资料）"
    resp = llm.invoke([
        SystemMessage(content=_GROUNDED_SYSTEM),
        HumanMessage(content=f"【检索资料】\n{ctx}\n\n【用户问题】\n{query}"),
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _generate_persona(llm, query: str, contexts: List[str]) -> str:
    """周医生 persona 回复（复用生产 ZHOU_KNOWLEDGE_PROMPT_TEMPLATE 与安全包裹）。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from modules.intervention.persona import ZHOU_KNOWLEDGE_PROMPT_TEMPLATE
    from modules.prompt_guard import INSTRUCTION_HIERARCHY_SUFFIX, wrap_user_text

    knowledge = "\n".join(contexts) if contexts else "（未检索到相关知识）"
    values = {
        "primary_emotion": "mixed",
        "intensity": "0.5",
        "risk": "0.2",
        "conversation_history": "",
        "retrieved_knowledge": knowledge,
        "probed_dimensions_note": "（尚未探测任何维度）",
        "assessor_context": "（无评估上下文，这是第一轮对话）",
        "phase": "check_in",
        "zhou_style_refs": "",
    }
    system_text = ZHOU_KNOWLEDGE_PROMPT_TEMPLATE.format(**values) + INSTRUCTION_HIERARCHY_SUFFIX
    resp = llm.invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=wrap_user_text(query)),
    ])
    return resp.content if isinstance(resp.content, str) else str(resp.content)


# ════════════════════════════════════════════════════════════════════
#  5. 评测（RAGAS 双 judge）
# ════════════════════════════════════════════════════════════════════

async def _safe_metric(name: str, coro_factory: Callable, attempts: int = 2) -> Optional[float]:
    """带重试的指标调用，失败返回 None。"""
    for attempt in range(attempts):
        try:
            res = await coro_factory()
            return float(res.value) if res is not None else None
        except Exception as e:
            logger.warning("%s 第 %d 次失败: %s", name, attempt + 1, e)
            await asyncio.sleep(1)
    return None


class GenerationEvaluator:
    def __init__(self, retriever: Any, smoke: bool = False):
        # 生成/改写（qwen3.7-max，关思考）
        self._generator = _make_langchain(settings.MODEL_NAME, 0.7, 1024, thinking_off=True)
        self._grounded_llm = _make_langchain(settings.MODEL_NAME, 0.0, 1024, thinking_off=True)
        self._retriever = retriever
        # judge：Run A 独立（flash），Run B 自评（max）
        self._judges = {
            "flash": _make_ragas_judge(settings.SCORING_MODEL_NAME),
            "max": _make_ragas_judge(settings.MODEL_NAME),
        }
        self._embed = ZhouEmbedder(QianwenEmbedding())

    async def score_query(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query, reference = row["query"], row["reference_answer"]

        # ── 检索 ──
        try:
            contexts = self._retriever.retrieve(query, top_k=3)
        except Exception as e:
            logger.warning("检索失败 query=%s: %s", query[:20], e)
            contexts = []

        # ── 生成 ──
        grounded, persona = "", ""
        try:
            grounded = _generate_grounded(self._grounded_llm, query, contexts)
        except Exception as e:
            logger.warning("grounded 生成失败: %s", e)
        try:
            persona = _generate_persona(self._generator, query, contexts)
        except Exception as e:
            logger.warning("persona 生成失败: %s", e)

        # ── RAGAS 打分 ──
        scores: Dict[str, Any] = {}
        for judge_name, judge in self._judges.items():
            if grounded and contexts:
                scores[f"faithfulness_{judge_name}"] = await _safe_metric(
                    f"faithfulness-{judge_name}",
                    lambda j=judge: Faithfulness(llm=j).ascore(
                        user_input=query, response=grounded, retrieved_contexts=contexts))
            if contexts:
                scores[f"context_relevance_{judge_name}"] = await _safe_metric(
                    f"context_relevance-{judge_name}",
                    lambda j=judge: ContextRelevance(llm=j).ascore(
                        user_input=query, retrieved_contexts=contexts))
        if persona and reference:
            scores["semantic_similarity"] = await _safe_metric(
                "semantic_similarity",
                lambda: SemanticSimilarity(embeddings=self._embed).ascore(
                    reference=reference, response=persona))

        return {
            "id": row["id"],
            "source_file": row["source_file"],
            "query": query,
            "reference_answer": reference,
            "contexts": contexts,
            "answer_grounded": grounded,
            "answer_persona": persona,
            "scores": scores,
        }


# ════════════════════════════════════════════════════════════════════
#  6. 报告
# ════════════════════════════════════════════════════════════════════

_METRIC_LABELS = {
    "faithfulness_flash": "忠实度 (独立 judge=flash)",
    "faithfulness_max": "忠实度 (自评 judge=max)",
    "context_relevance_flash": "上下文相关性 (独立 judge=flash)",
    "context_relevance_max": "上下文相关性 (自评 judge=max)",
    "semantic_similarity": "医生金标准相似度 (bge-m3)",
}


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, var ** 0.5


def _write_report(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    n = len(results)
    # ── 聚合 ──
    metrics = list(_METRIC_LABELS.keys())
    agg: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        values = [r["scores"].get(m) for r in results]
        mean, std = _mean_std(values)
        agg[m] = {"mean": mean, "std": std, "n": sum(1 for v in values if v is not None)}

    # ── 双 judge 分歧 ──
    diff_f, diff_c = [], []
    for r in results:
        f_flash, f_max = r["scores"].get("faithfulness_flash"), r["scores"].get("faithfulness_max")
        c_flash, c_max = r["scores"].get("context_relevance_flash"), r["scores"].get("context_relevance_max")
        if f_flash is not None and f_max is not None:
            diff_f.append(abs(f_flash - f_max))
        if c_flash is not None and c_max is not None:
            diff_c.append(abs(c_flash - c_max))
    flagged = [r["id"] for r in results
               if (r["scores"].get("faithfulness_flash") is not None
                   and r["scores"].get("faithfulness_max") is not None
                   and abs(r["scores"]["faithfulness_flash"] - r["scores"]["faithfulness_max"]) > 0.15)
               or (r["scores"].get("context_relevance_flash") is not None
                   and r["scores"].get("context_relevance_max") is not None
                   and abs(r["scores"]["context_relevance_flash"] - r["scores"]["context_relevance_max"]) > 0.15)]

    # ── Markdown ──
    lines = []
    lines.append("# RAG 生成侧评测结果（RAGAS 0.4.3）")
    lines.append("")
    lines.append(f"- 测试集：周医生真实门诊语料 **{n}** 条（脱敏），医生回应即金标准")
    lines.append(f"- 检索模式：`{config['retriever']}`（QR→BM25 / 生产管线 BGE 重排）")
    lines.append(f"- 生成模型：{settings.MODEL_NAME}（grounded-QA + 周医生 persona 各一次，关思考）")
    lines.append(f"- 独立 judge：{settings.SCORING_MODEL_NAME}（Run A）｜自评 judge：{settings.MODEL_NAME}（Run B）")
    lines.append("")
    lines.append("## 指标聚合（均值 ± 标准差）")
    lines.append("")
    lines.append("| 指标 | 均值 ± 标准差 | 有效数 |")
    lines.append("|---|---|---|")
    for m in metrics:
        mean, std = agg[m]["mean"], agg[m]["std"]
        cell = "--" if mean is None else f"{mean:.3f} ± {std:.3f}"
        lines.append(f"| {_METRIC_LABELS[m]} | {cell} | {agg[m]['n']} |")
    lines.append("")
    if diff_f or diff_c:
        lines.append("## 双 judge 分歧")
        lines.append("")
        lines.append(f"- 忠实度 |Δ|：均值 **{sum(diff_f)/len(diff_f):.3f}**（n={len(diff_f)}）")
        lines.append(f"- 上下文相关性 |Δ|：均值 **{sum(diff_c)/len(diff_c):.3f}**（n={len(diff_c)}）")
        lines.append(f"- 任一指标分歧 > 0.15 的查询：**{len(flagged)}** 条 → {', '.join(flagged) if flagged else '无'}")
        lines.append("")

    lines.append("## 逐条明细")
    lines.append("")
    for r in results:
        s = r["scores"]
        fmt = lambda k: "--" if s.get(k) is None else f"{s[k]:.3f}"
        lines.append(f"**{r['id']}** ({r['source_file']})")
        lines.append(f"- 患者：{r['query'][:80]}")
        lines.append(f"- 忠实度 flash={fmt('faithfulness_flash')} max={fmt('faithfulness_max')}｜"
                     f"上下文相关性 flash={fmt('context_relevance_flash')} max={fmt('context_relevance_max')}｜"
                     f"金标准相似度={fmt('semantic_similarity')}")
        if r["answer_grounded"]:
            lines.append(f"- grounded 回答：{r['answer_grounded'][:120]}")
        lines.append("")
    report_md = "\n".join(lines)

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write(report_md)
    summary = {
        "config": config,
        "n_queries": n,
        "aggregate": {m: agg[m] for m in metrics},
        "disagreement": {
            "faithfulness_mean_abs_diff": (sum(diff_f) / len(diff_f)) if diff_f else None,
            "context_relevance_mean_abs_diff": (sum(diff_c) / len(diff_c)) if diff_c else None,
            "flagged_queries": flagged,
        },
        "queries": results,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


# ════════════════════════════════════════════════════════════════════
#  main
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAG 生成侧评测（RAGAS）")
    parser.add_argument("--build-test-set", action="store_true", help="只构建测试集")
    parser.add_argument("-n", "--num", type=int, default=40, help="测试集条数（默认 40）")
    parser.add_argument("--force", action="store_true", help="强制重建测试集")
    parser.add_argument("--retriever", choices=["fast", "production"], default="fast")
    parser.add_argument("--smoke", action="store_true", help="5 条冒烟")
    args = parser.parse_args()

    if args.build_test_set:
        sample = build_test_set(args.num, force=args.force)
        print(f"测试集已构建：{len(sample)} 条 → {TEST_SET_PATH}")
        print("\n--- 示例 ---")
        for s in sample[:3]:
            print(f"[患者] {s['query'][:60]}")
            print(f"[医生] {s['reference_answer'][:70]}")
            print()
        return

    sample = build_test_set(args.num, force=args.force)
    num = 5 if args.smoke else len(sample)
    sample = sample[:num]

    # 改写器（共享一个实例）
    from core.rag.query_rewriter import QueryRewriter
    rewriter_llm = OpenAICompatibleAdapter(config=LLMConfig(
        model_name=settings.REWRITER_MODEL_NAME, temperature=0, max_tokens=256,
        streaming=False, timeout=60,
        model_kwargs={"extra_body": {"enable_thinking": False}},
    )).llm
    rewriter = QueryRewriter(rewriter_llm)
    if args.retriever == "production":
        retriever = ProductionRetriever(rewriter=rewriter)
    else:
        retriever = FastRetriever(rewriter=rewriter)

    print("=" * 70)
    print("RAG 生成侧评测（RAGAS 0.4.3）")
    print(f"retriever={args.retriever} | n={num} | 生成={settings.MODEL_NAME} | "
          f"独立judge={settings.SCORING_MODEL_NAME} | 自评judge={settings.MODEL_NAME}")
    print("=" * 70)

    evaluator = GenerationEvaluator(retriever=retriever, smoke=args.smoke)

    async def _run_all(evaluator, sample) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for i, row in enumerate(sample, 1):
            t0 = time.time()
            try:
                r = await evaluator.score_query(row)
                results.append(r)
                s = r["scores"]
                print(f"[{i}/{num}] {r['id']} "
                      f"faith_f={s.get('faithfulness_flash')} max={s.get('faithfulness_max')} "
                      f"ctx_f={s.get('context_relevance_flash')} max={s.get('context_relevance_max')} "
                      f"sim={s.get('semantic_similarity')} | {time.time()-t0:.0f}s", flush=True)
            except Exception as e:
                logger.exception("查询 %s 整体失败", row["id"])
                print(f"[{i}/{num}] {row['id']} 失败: {e}", flush=True)
        return results

    t_start = time.time()
    # 单事件循环跑完全部查询（AsyncOpenAI/httpx 客户端绑定单 loop，不能每个查询新建 loop）
    results = asyncio.run(_run_all(evaluator, sample))
    summary = _write_report(results, {"retriever": args.retriever})
    print("\n" + "=" * 70)
    print("汇总（均值）：")
    for m in ["faithfulness_flash", "faithfulness_max", "context_relevance_flash",
              "context_relevance_max", "semantic_similarity"]:
        a = summary["aggregate"][m]
        cell = "--" if a["mean"] is None else f"{a['mean']:.3f} ± {a['std']:.3f}"
        print(f"  {_METRIC_LABELS[m]:<28s} {cell}  (n={a['n']})")
    print(f"双 judge 分歧：忠实度 |Δ|={summary['disagreement']['faithfulness_mean_abs_diff']}，"
          f"相关性 |Δ|={summary['disagreement']['context_relevance_mean_abs_diff']}")
    print(f"分歧 > 0.15 的查询数：{len(summary['disagreement']['flagged_queries'])}")
    print(f"总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    print(f"报告：{RESULTS_MD}\nJSON：{RESULTS_JSON}")


if __name__ == "__main__":
    main()
