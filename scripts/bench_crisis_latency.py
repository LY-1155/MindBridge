"""危机响应路径延迟基准（S3）——「高危场景安全短路零延迟直达」的实测背书

口径（重要，面试会被追问）：
- 危机**响应** = 检测裁决后的确定性模板阶段（CrisisHandler.handle），不经 LLM/网络。
  这是简历「热线号码不经 LLM」「零延迟直达」的精确所指。
- 危机**端到端** = 检测（LLM 语义安全评估器，intervene() 内 _run_doctor_assessment）
  + 响应。检测有 LLM 延迟，不在本基准声称范围内——基准只测响应阶段，并在 B 中
  隔离检测（模拟「已裁决 crisis」）。

测量三部分：
  A. CrisisHandler.handle() 延迟                —— 纯确定性响应（头条数字，n=500）
  B. intervene() crisis 分支（真实 session：Redis 读 + 内存追加；USE_DATABASE=false 未含 MySQL 写）
     —— 检测后响应的完整路由（n=200）
  C. 对照：普通 LLM 生成（generate_general）       —— 量化「短路」省掉的秒级延迟

运行：& "D:\Anaconda\envs\emotion\python.exe" -u scripts\bench_crisis_latency.py [--skip-llm] [--n-a 500] [--n-b 200]
输出：stdout + scripts/crisis_latency_results.{md,json}
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# ── logging：与生产同 level（WARNING，trigger 的告警日志会被格式化），
#    但丢弃输出，避免刷屏；格式化成本计入测量（% 插值发生在记录创建时）。──
class _DropHandler(logging.Handler):
    def emit(self, record):  # noqa: A003
        pass


def _setup_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    root.addHandler(_DropHandler())


from schemas.contracts import InterventionRequest  # noqa: E402
from modules.intervention.crisis_handler import CrisisHandler  # noqa: E402
from modules.intervention.service import InterventionService  # noqa: E402

RESULTS_MD = PROJECT_ROOT / "scripts" / "crisis_latency_results.md"
RESULTS_JSON = PROJECT_ROOT / "scripts" / "crisis_latency_results.json"


def _make_crisis_req(session_id: str) -> InterventionRequest:
    return InterventionRequest(
        user_text="我真的不想活了，想自杀",
        route={"route": "crisis", "reason": "高危关键词匹配", "confidence": 0.98},
        emotion={"primary_emotion": "distress", "intensity": 0.95, "risk": 0.98},
        safety={"level": 2, "blocked": False, "matched_terms": ["自杀", "不想活"]},
        safety_verdict={"verdict": "crisis"},
        session_id=session_id,
        user_id="bench-user",
    )


def _make_general_req(session_id: str) -> InterventionRequest:
    return InterventionRequest(
        user_text="最近总是睡不好，白天上课很累，有什么办法吗",
        route={"route": "general", "reason": "睡眠困扰", "confidence": 0.9},
        emotion={"primary_emotion": "tired", "intensity": 0.6, "risk": 0.1},
        safety={"level": 0, "blocked": False, "matched_terms": []},
        safety_verdict=None,
        session_id=session_id,
        user_id="bench-user",
    )


def _percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(xs) else f
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _report(label: str, times_ms: list[float]) -> dict:
    times_ms = sorted(times_ms)
    agg = {
        "n": len(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": _percentile(times_ms, 50),
        "p95_ms": _percentile(times_ms, 95),
        "p99_ms": _percentile(times_ms, 99),
        "max_ms": max(times_ms),
    }
    print(f"\n{label}  (n={agg['n']})")
    print(
        f"  mean={agg['mean_ms']:.3f}ms  "
        f"p50={agg['p50_ms']:.3f}ms  p95={agg['p95_ms']:.3f}ms  "
        f"p99={agg['p99_ms']:.3f}ms  max={agg['max_ms']:.3f}ms"
    )
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="跳过 C 对照（LLM 生成）")
    ap.add_argument("--n-a", type=int, default=500)
    ap.add_argument("--n-b", type=int, default=200)
    args = ap.parse_args()
    _setup_logging()

    print("=" * 72)
    print("危机响应延迟基准（S3）")
    print("口径：响应 = 检测裁决后的确定性模板阶段（不经 LLM/网络）；")
    print("      检测（LLM 语义评估器）有 LLM 延迟，不在声称范围。")
    print("=" * 72)

    results: dict = {"config": {"n_a": args.n_a, "n_b": args.n_b, "skip_llm": args.skip_llm}}

    # ── A. CrisisHandler.handle()（真实 EmergencyPushService，dry-run 配置）──
    handler = CrisisHandler()
    ta: list[float] = []
    last = None
    for i in range(args.n_a):
        req = _make_crisis_req(f"bench-a-{i:04d}")
        t0 = time.perf_counter()
        last = handler.handle(req)
        ta.append((time.perf_counter() - t0) * 1000)
    if not last or not last.emergency_triggered:
        print("  [!] A 未触发（冷却短路？），检查实现")
    results["A_handle"] = _report("A. 危机响应生成 CrisisHandler.handle()（不经 LLM/网络）", ta)

    # ── B. intervene() crisis 分支（真实 session + Redis 落库，隔离检测）──
    from core.memory.session_memory import SessionManager

    svc = InterventionService(crisis_handler=CrisisHandler())
    # 隔离检测：模拟「已裁决 crisis」——测响应路由真实执行（session 查找 + handle + 落库）
    svc._run_doctor_assessment = lambda req, session, route: ("crisis", None, None)  # noqa: SLF001
    tb: list[float] = []
    for i in range(args.n_b):
        sid = SessionManager.create_session("bench-user")
        req = _make_crisis_req(sid)
        t0 = time.perf_counter()
        try:
            svc.intervene(req)
            tb.append((time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            if len(tb) < 1 and i < 3:
                print(f"  [!] B iter{i} failed: {e}")
    results["B_route"] = _report(
        "B. 完整危机路由 intervene() crisis 分支（含 session，已裁决后；USE_DATABASE=false 未含 MySQL 写）", tb
    )

    # ── C. 对照：普通 LLM 生成（没有短路时这条回复要走 LLM 的成本）──
    if not args.skip_llm:
        try:
            from core.llm.base import get_llm_adapter
            from modules.intervention.generator import InterventionReplyGenerator

            llm = get_llm_adapter("qwen")
            gen = InterventionReplyGenerator(llm=llm)
            tc: list[float] = []
            for i in range(2):
                req_c = _make_general_req("bench-c-ctrl")
                t0 = time.perf_counter()
                try:
                    gen.generate_general(req_c)
                    tc.append((time.perf_counter() - t0) * 1000)
                except Exception as e:  # noqa: BLE001
                    print(f"  [!] C 对照生成失败: {e}")
            results["C_llm_contrast"] = _report("C. 对照：LLM 生成一条正常回复（generate_general）", tc)
            if results["C_llm_contrast"] and results["A_handle"]:
                mean_a = results["A_handle"]["mean_ms"]
                mean_c = results["C_llm_contrast"]["mean_ms"]
                ratio = mean_c / max(mean_a, 1e-6)
                results["short_circuit_ratio"] = ratio
                print(f"\n短路对比：危机响应 {mean_a:.2f}ms vs LLM 生成 {mean_c:.0f}ms "
                      f"≈ {ratio:,.0f}x")
        except Exception as e:  # noqa: BLE001
            print(f"  [C 对照跳过] LLM 不可用: {e}")
            results["C_llm_contrast"] = None

    # ── 存档 ──
    RESULTS_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 危机响应延迟基准（S3）",
        "",
        "口径：响应 = 检测裁决后的确定性模板阶段（CrisisHandler.handle），不经 LLM/网络；",
        "检测（LLM 语义安全评估器）有 LLM 延迟，不在本基准声称范围。",
        "",
        "| 测量 | mean | p50 | p95 | p99 | max | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in [
        ("A_handle", "A 危机响应生成 handle()"),
        ("B_route", "B 完整危机路由（已裁决后，未含 MySQL 写）"),
        ("C_llm_contrast", "C 对照 LLM 生成"),
    ]:
        agg = results.get(key)
        if not agg:
            continue
        lines.append(
            f"| {label} | {agg['mean_ms']:.2f}ms | {agg['p50_ms']:.2f}ms | "
            f"{agg['p95_ms']:.2f}ms | {agg['p99_ms']:.2f}ms | {agg['max_ms']:.2f}ms | {agg['n']} |"
        )
    if results.get("short_circuit_ratio"):
        lines.append("")
        lines.append(f"- 短路对比：危机响应 vs LLM 生成 ≈ **{results['short_circuit_ratio']:,.0f}x**")
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n存档：{RESULTS_MD.name} / {RESULTS_JSON.name}")


if __name__ == "__main__":
    main()
