#!/usr/bin/env python
"""风险公式参数校准工具。

读取 config/calibration_samples.json 中的标注样本，
对 router_rules.json 中的 risk_formula 和 thresholds 做 grid search，
输出最优参数组合。避免手动盲调。

用法:
  python scripts/calibrate_risk.py                # 默认参数范围 grid search
  python scripts/calibrate_risk.py --write-best     # 搜索后直接写回 router_rules.json
  python scripts/calibrate_risk.py --fear 0.45,0.50,0.55  # 只搜指定参数值

样本格式 (calibration_samples.json):
  {"text": "...", "primary_emotion": "anxiety", "intensity": 0.77,
   "safety_level": 0, "expected_route": "knowledge"}
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 解决 Windows GBK 终端 Unicode 输出问题
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLES_PATH = ROOT / "config" / "calibration_samples.json"
DEFAULT_RULES_PATH = ROOT / "config" / "router_rules.json"

Route = str  # "general" | "comfort" | "knowledge" | "crisis"


# ── 路由逻辑（与 router_service.py 完全一致）─────────────────────

def risk_to_route(risk: float, crisis_t: float, comfort_t: float, knowledge_t: float) -> Route:
    if risk >= crisis_t:
        return "crisis"
    if risk >= comfort_t:
        return "comfort"
    if risk >= knowledge_t:
        return "knowledge"
    return "general"


def escalate_band(route: Route) -> Route:
    if route == "general":
        return "knowledge"
    if route == "knowledge":
        return "comfort"
    if route == "comfort":
        return "crisis"
    return route


def should_escalate_safety(safety_level: int, rules: dict) -> bool:
    esc = rules.get("safety_escalation", {})
    return bool(esc.get("level_1_escalate")) and safety_level >= 2


# ── 风险公式（与 stub.py 完全一致）───────────────────────────────

def compute_risk(
    primary_emotion: str,
    intensity: float,
    safety_level: int,
    risk_cfg: dict,
) -> float:
    base = risk_cfg["emotion_base"].get(primary_emotion, 0.0)
    iw = float(risk_cfg["intensity_weight"])
    sw = float(risk_cfg["safety_weight"])
    return round(min(base + intensity * iw + safety_level * sw, 1.0), 2)


def predict_route(
    sample: dict,
    risk_cfg: dict,
    thresholds: dict,
    rules: dict,
) -> Route:
    """给定样本和全部配置，预测最终路由。"""
    risk = compute_risk(
        sample["primary_emotion"],
        sample["intensity"],
        sample["safety_level"],
        risk_cfg,
    )
    route = risk_to_route(
        risk,
        float(thresholds["crisis_risk"]),
        float(thresholds["comfort_risk"]),
        float(thresholds["knowledge_risk"]),
    )
    if should_escalate_safety(sample["safety_level"], rules):
        route = escalate_band(route)
    return route


# ── 评估 ────────────────────────────────────────────────────────

def evaluate(
    samples: List[dict],
    risk_cfg: dict,
    thresholds: dict,
    rules: dict,
) -> Tuple[float, List[dict]]:
    """返回 (accuracy, per_sample_details)。"""
    correct = 0
    details = []
    for s in samples:
        pred = predict_route(s, risk_cfg, thresholds, rules)
        ok = pred == s["expected_route"]
        if ok:
            correct += 1
        details.append({**s, "predicted_route": pred, "risk": compute_risk(
            s["primary_emotion"], s["intensity"], s["safety_level"], risk_cfg), "correct": ok})
    return correct / len(samples) if samples else 0.0, details


# ── Grid search ─────────────────────────────────────────────────

def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",")]


def build_search_space(args) -> Dict[str, List[float]]:
    """根据命令行参数构建搜索空间。"""
    return {
        "fear_base": parse_float_list(args.fear) if args.fear else [0.45, 0.50, 0.55, 0.60],
        "anxiety_base": parse_float_list(args.anxiety) if args.anxiety else [0.30, 0.35, 0.40, 0.45],
        "anger_base": parse_float_list(args.anger) if args.anger else [0.40, 0.45, 0.50],
        "stress_base": parse_float_list(args.stress) if args.stress else [0.40, 0.45, 0.50],
        "sadness_base": parse_float_list(args.sadness) if args.sadness else [0.30, 0.35, 0.40],
        "intensity_weight": parse_float_list(args.intensity_weight) if args.intensity_weight else [0.10, 0.15, 0.20, 0.25],
        "safety_weight": parse_float_list(args.safety_weight) if args.safety_weight else [0.10, 0.15, 0.20],
        "crisis_t": parse_float_list(args.crisis) if args.crisis else [0.70, 0.75, 0.80, 0.85],
        "comfort_t": parse_float_list(args.comfort) if args.comfort else [0.35, 0.40, 0.45, 0.50],
        "knowledge_t": parse_float_list(args.knowledge) if args.knowledge else [0.10, 0.15, 0.20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="风险公式参数校准")
    parser.add_argument("--samples", type=str, default=str(DEFAULT_SAMPLES_PATH))
    parser.add_argument("--rules", type=str, default=str(DEFAULT_RULES_PATH))
    parser.add_argument("--write-best", action="store_true", help="将最优参数写回 router_rules.json")
    parser.add_argument("--top", type=int, default=10, help="显示前 N 个最优组合 (default 10)")

    # 各参数搜索范围（逗号分隔）
    parser.add_argument("--fear", type=str, help="fear base 值列表，如 0.45,0.50,0.55")
    parser.add_argument("--anxiety", type=str)
    parser.add_argument("--anger", type=str)
    parser.add_argument("--stress", type=str)
    parser.add_argument("--sadness", type=str)
    parser.add_argument("--intensity-weight", type=str, dest="intensity_weight")
    parser.add_argument("--safety-weight", type=str, dest="safety_weight")
    parser.add_argument("--crisis", type=str, help="crisis 阈值列表，如 0.70,0.75,0.80")
    parser.add_argument("--comfort", type=str, help="comfort 阈值列表，如 0.35,0.40,0.45")
    parser.add_argument("--knowledge", type=str, help="knowledge 阈值列表，如 0.10,0.15,0.20")
    args = parser.parse_args()

    # 加载
    samples_data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples: List[dict] = samples_data["samples"]
    rules = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    orig_risk_cfg = rules.get("risk_formula", {})
    orig_thresholds = rules.get("thresholds", {})

    if not samples:
        print("[ERROR] 校准样本为空，请先在 calibration_samples.json 中添加样本")
        sys.exit(1)

    # 基线
    baseline_acc, baseline_details = evaluate(samples, orig_risk_cfg, orig_thresholds, rules)
    wrong_baseline = [d for d in baseline_details if not d["correct"]]

    print("=" * 72)
    print(f"{'基线评估':^72}")
    print("=" * 72)
    print(f"  样本数: {len(samples)}")
    print(f"  准确率: {baseline_acc:.1%} ({int(baseline_acc * len(samples))}/{len(samples)})")
    if wrong_baseline:
        print(f"\n  当前参数下的错误样本:")
        for d in wrong_baseline:
            print(f"    [X] \"{d['text'][:40]}...\"")
            print(f"      emotion={d['primary_emotion']} intensity={d['intensity']} "
                  f"safety={d['safety_level']} risk={d['risk']}")
            print(f"      expected={d['expected_route']} predicted={d['predicted_route']}")
    print()

    # Grid search
    space = build_search_space(args)
    keys = list(space.keys())
    total = 1
    for v in space.values():
        total *= len(v)
    print(f"  搜索空间: {len(keys)} 个维度, 共 {total} 组组合\n")

    best_acc = 0.0
    best_configs: List[Tuple[float, dict]] = []

    for combo in product(*space.values()):
        # 构建 risk_cfg
        risk_cfg: Dict[str, Any] = {
            "emotion_base": {
                "fear": combo[keys.index("fear_base")] if "fear_base" in keys else orig_risk_cfg["emotion_base"]["fear"],
                "anxiety": combo[keys.index("anxiety_base")] if "anxiety_base" in keys else orig_risk_cfg["emotion_base"]["anxiety"],
                "anger": combo[keys.index("anger_base")] if "anger_base" in keys else orig_risk_cfg["emotion_base"]["anger"],
                "stress": combo[keys.index("stress_base")] if "stress_base" in keys else orig_risk_cfg["emotion_base"]["stress"],
                "sadness": combo[keys.index("sadness_base")] if "sadness_base" in keys else orig_risk_cfg["emotion_base"]["sadness"],
                "confusion": orig_risk_cfg["emotion_base"]["confusion"],
                "happiness": orig_risk_cfg["emotion_base"]["happiness"],
                "neutral": orig_risk_cfg["emotion_base"]["neutral"],
                "distress": orig_risk_cfg["emotion_base"]["distress"],
            },
            "intensity_weight": combo[keys.index("intensity_weight")] if "intensity_weight" in keys else orig_risk_cfg["intensity_weight"],
            "safety_weight": combo[keys.index("safety_weight")] if "safety_weight" in keys else orig_risk_cfg["safety_weight"],
        }

        thresholds = dict(orig_thresholds)
        if "crisis_t" in keys:
            thresholds["crisis_risk"] = combo[keys.index("crisis_t")]
        if "comfort_t" in keys:
            thresholds["comfort_risk"] = combo[keys.index("comfort_t")]
        if "knowledge_t" in keys:
            thresholds["knowledge_risk"] = combo[keys.index("knowledge_t")]

        acc, _ = evaluate(samples, risk_cfg, thresholds, rules)

        if acc > best_acc:
            best_acc = acc
            best_configs = [(acc, {"risk_formula": risk_cfg, "thresholds": thresholds})]
        elif acc == best_acc:
            best_configs.append((acc, {"risk_formula": risk_cfg, "thresholds": thresholds}))

    # 排序：准确率优先，再按 crisis 阈值较低（更保守）的排前
    best_configs.sort(key=lambda x: (-x[0], x[1]["thresholds"]["crisis_risk"]))

    print("=" * 72)
    print(f"{'Top-%d 最优组合' % min(args.top, len(best_configs)):^72}")
    print("=" * 72)

    for rank, (acc, cfg) in enumerate(best_configs[:args.top], 1):
        rf = cfg["risk_formula"]
        th = cfg["thresholds"]
        eb = rf["emotion_base"]
        print(f"\n  #{rank}  准确率: {acc:.1%}")
        print(f"  | threshold:  crisis={th.get('crisis_risk','?')}  "
              f"comfort={th.get('comfort_risk','?')}  knowledge={th.get('knowledge_risk','?')}")
        print(f"  | emotion_base: fear={eb['fear']}  anxiety={eb['anxiety']}  "
              f"anger={eb['anger']}  sadness={eb['sadness']}  stress={eb['stress']}")
        print(f"  | weights: intensity={rf.get('intensity_weight','?')}  "
              f"safety={rf.get('safety_weight','?')}")

        # 详细预测
        _, details = evaluate(samples, rf, th, rules)
        for d in details:
            status = "OK" if d["correct"] else "XX"
            print(f"     {status} {d['expected_route']:>8} ← risk={d['risk']:.2f}  "
                  f"\"{d['text'][:50]}\"")

    print(f"\n  共 {len(best_configs)} 组达到最优准确率 {best_acc:.1%}")

    # 写回
    if args.write_best:
        best_cfg = best_configs[0][1]
        rules["risk_formula"] = best_cfg["risk_formula"]
        rules["thresholds"] = best_cfg["thresholds"]
        rules_path = Path(args.rules)
        rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[OK] 最优配置已写入 {rules_path}")


if __name__ == "__main__":
    main()
