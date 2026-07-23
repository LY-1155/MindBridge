"""验证三个路由提示词的内容完整性 + 结构合理性。

运行方式: python scripts/validate_prompts.py
目的: 确保提示词改动后 (1) 占位符一致 (2) 关键规则项全部存在 (3) 格式化不抛异常
"""

from __future__ import annotations
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.intervention.generator import (
    COMFORT_SYSTEM_PROMPT,
    KNOWLEDGE_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
)


# ── 1. 占位符一致性检查 ──────────────────────────────────────
REQUIRED_PLACEHOLDERS = [
    "{primary_emotion}",
    "{intensity}",
    "{risk}",
    "{conversation_history}",
]


def check_placeholders(name: str, prompt: str, required: list[str]) -> list[str]:
    missing = [p for p in required if p not in prompt]
    return missing


# ── 2. 每条提示词必须包含的关键规则 ──────────────────────────
# 对应医生的两条核心反馈 + 实际对话测试反馈
EMPATHY_CHECKS = [
    ("消化后再表达，不重复原话", "消化了再表达"),
    ("口语化，不书面", "像朋友"),
    ("避免书面加强语势", "书面"),
    ("不是在填问卷", "填问卷"),
    ("探测式共情——共情与探测融为一体", "探测式共情"),
]

LENGTH_CHECKS = [
    ("COMFORT: 不说教，不罗列", "探测式共情"),
    ("KNOWLEDGE: ≤4 句话", "不超过 4 句话"),
    ("GENERAL: 2-3 句话", "2-3 句话"),
]

EXAMPLE_CHECKS = [
    ("正面示例 (✓)", "✓"),
    ("反面示例 (✗)", "✗"),
    ("每个示例有解释", "→ 为什么"),
]

# 路由特有检查
ROUTE_SPECIFIC = {
    "COMFORT": [
        ("不以解决问题为目标", "不分析"),
        ("探测式共情——共情+探测一体", "共情（"),
        ("情绪浓时不探测，只承接", "我听到你"),
    ],
    "KNOWLEDGE": [
        ("检索内容三级可信度判断", "高度匹配"),
        ("先共情再讲知识", "先共情"),
        ("诚实地说不懂", "不够多"),
    ],
    "GENERAL": [
        ("情绪信号识别", "情绪信号"),
        ("轻松过渡而非切咨询师模式", "想聊聊吗"),
    ],
}


def run_checks():
    prompts = {
        "COMFORT": COMFORT_SYSTEM_PROMPT,
        "KNOWLEDGE": KNOWLEDGE_SYSTEM_PROMPT,
        "GENERAL": GENERAL_SYSTEM_PROMPT,
    }

    all_ok = True

    # ── 占位符检查 ──
    print("=" * 60)
    print("1. 占位符一致性检查")
    print("=" * 60)
    for name, prompt in prompts.items():
        missing = check_placeholders(name, prompt, REQUIRED_PLACEHOLDERS)
        # KNOWLEDGE 额外有 retrieved_knowledge
        if name == "KNOWLEDGE":
            if "{retrieved_knowledge}" not in prompt:
                missing.append("{retrieved_knowledge}")
        if missing:
            print(f"  ❌ {name}: 缺少占位符 {missing}")
            all_ok = False
        else:
            extra = ""
            if name == "KNOWLEDGE":
                extra = " (+ retrieved_knowledge)"
            print(f"  ✅ {name}: 占位符完整{extra}")

    # ── 共情深度检查 ──
    print("\n" + "=" * 60)
    print('2. 共情深度检查（医生反馈 #1：让人感到“你真的在听”）')
    print("=" * 60)
    for label, keyword in EMPATHY_CHECKS:
        results = []
        for name, prompt in prompts.items():
            if keyword in prompt:
                results.append(name)
        if results:
            print(f"  ✅ [{label}] → 覆盖: {', '.join(results)}")
        else:
            print(f"  ❌ [{label}] → 未覆盖任何路由!")
            all_ok = False

    # ── 回复长度检查 ──
    print("\n" + "=" * 60)
    print('3. 回复长度约束检查（医生反馈 #2：追求“持续对话”）')
    print("=" * 60)
    for label, keyword in LENGTH_CHECKS:
        route = label.split(":")[0].strip()
        prompt = prompts.get(route, "")
        if keyword in prompt:
            print(f"  ✅ [{label}]")
        else:
            print(f"  ❌ [{label}] — 未找到 '{keyword}'")
            all_ok = False

    # ── 示例质量检查 ──
    print("\n" + "=" * 60)
    print("4. Few-Shot 示例完整性")
    print("=" * 60)
    for name, prompt in prompts.items():
        has_good = "✓" in prompt
        has_bad = "✗" in prompt
        has_why = "→ 为什么" in prompt or "-> 为什么" in prompt
        good_count = prompt.count("✓")
        bad_count = prompt.count("✗")
        if has_good and has_bad and has_why:
            print(f"  ✅ {name}: {good_count}个好示例, {bad_count}个坏示例, 各有解释")
        else:
            print(f"  ⚠️  {name}: good={has_good}, bad={has_bad}, why={has_why}")
            all_ok = False

    # ── 路由特有检查 ──
    print("\n" + "=" * 60)
    print("5. 路由特有规则")
    print("=" * 60)
    for route, checks in ROUTE_SPECIFIC.items():
        prompt = prompts[route]
        for label, keyword in checks:
            if keyword in prompt:
                print(f"  ✅ [{route}] {label}")
            else:
                print(f"  ❌ [{route}] {label} — 未找到 '{keyword}'")
                all_ok = False

    # ── 格式化干跑 ──
    print("\n" + "=" * 60)
    print("6. 格式化干跑（占位符替换不抛异常）")
    print("=" * 60)
    mock_values = {
        "primary_emotion": "anxiety",
        "intensity": "0.7",
        "risk": "0.3",
        "conversation_history": "用户：今天心情不太好\n助手：愿意聊聊吗？",
        "retrieved_knowledge": "[检索结果] CBT 灾难化思维干预方案",
    }
    for name, prompt in prompts.items():
        try:
            # 只传该 prompt 需要的 key
            vals = dict(mock_values)
            rendered = prompt.format(**vals)
            # 检查渲染后没有残留的 {placeholder}
            if "{" in rendered:
                print(f"  ⚠️  {name}: 渲染后仍有未替换占位符")
                all_ok = False
            else:
                print(f"  ✅ {name}: 格式化成功 ({len(prompt)}→{len(rendered)} 字符)")
        except KeyError as e:
            print(f"  ❌ {name}: 格式化失败 — 缺少参数 {e}")
            all_ok = False
        except Exception as e:
            print(f"  ❌ {name}: 格式化失败 — {e}")
            all_ok = False

    # ── 总结 ──
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 全部检查通过，三个提示词改动有效。")
    else:
        print("❌ 存在未通过的检查项，请修复后重新验证。")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
