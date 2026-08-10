"""验证周医生风格参考库：检索质量 + 脱敏效果抽查

用法：
  python scripts/verify_zhou_style.py            # 检索 5 个典型情景
  python scripts/verify_zhou_style.py --privacy  # 额外抽查脱敏（找残留 PII）
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 典型青少年家庭场景 ───────────────────────────────────────
QUERIES = [
    "孩子不上学，天天在家打游戏",
    "我妈总逼我学习，压力好大",
    "跟男朋友分手了，好难受",
    "我睡不着，经常做噩梦",
    "感觉活着没意思，不想活了",
    "我爸我妈天天吵架",
]


def verify_retrieval():
    from modules.intervention.rag.zhou_style import get_zhou_style_retriever
    retriever = get_zhou_style_retriever()
    if not retriever.is_available():
        print("[错误] ZhouStyle 索引不可用（索引未构建？）")
        return False

    print("=== 检索质量抽查 ===\n")
    all_ok = True
    for q in QUERIES:
        hits = retriever.retrieve(q, top_k=3)
        print(f"查询: 「{q}」")
        if not hits:
            print("  ⚠️  无命中")
            all_ok = False
        for i, h in enumerate(hits, 1):
            print(f"  {i}. [{h['score']:.2f}] 患者:「{h['human'][:40]}」")
            print(f"     医生:「{h['doctor'][:60]}」")
        print()
    return all_ok


def check_privacy(n=200, seed=42):
    from core.privacy.desensitize import desensitize

    base = Path(__file__).resolve().parent.parent / "心理医生访谈数据"
    files = list(base.rglob("*.json"))
    random.seed(seed)
    random.shuffle(files)

    # PII 模式（脱敏后不应再出现）
    from core.privacy.desensitize import COMMON_SURNAMES

    phone_re = re.compile(r"1[3-9]\d{9}")
    long_digit_re = re.compile(r"\d{10,}")
    # 姓氏+职业称呼；排除"时候"里的"候"（姓氏字但非 PII）
    surname_role_re = re.compile(f"(?<!时)[{COMMON_SURNAMES}](?:医生|大夫|主任|院长|护士)")
    year_re = re.compile(r"20\d{2}年")

    checked = 0
    violations = []
    for p in files:
        if checked >= n:
            break
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in data:
            for turn in item.get("conversations", []):
                raw = turn.get("value", "")
                if not isinstance(raw, str) or len(raw) < 4:
                    continue
                cleaned = desensitize(raw)
                for label, pattern in [("手机号", phone_re), ("长数字", long_digit_re),
                                       ("姓名+职业", surname_role_re), ("年份", year_re)]:
                    m = pattern.search(cleaned)
                    if m:
                        ctx = cleaned[max(0, m.start() - 8):m.end() + 8]
                        violations.append((label, f"…{ctx}…"))
                        break
                checked += 1
                if checked >= n:
                    break
            if checked >= n:
                break

    print(f"=== 脱敏效果抽查：{checked} 轮发言 ===")
    if violations:
        print(f"⚠️  发现 {len(violations)} 处疑似 PII 残留：")
        for label, text in violations[:10]:
            print(f"  [{label}] {text}")
        return False
    print("✅ 未发现残留 PII（手机号/长数字/姓名+职业/年份）")
    return True


def main():
    parser = argparse.ArgumentParser(description="验证周医生风格参考库")
    parser.add_argument("--privacy", action="store_true", help="抽查脱敏效果")
    args = parser.parse_args()

    ok = verify_retrieval()
    if args.privacy:
        p_ok = check_privacy()
        ok = ok and p_ok

    print("=== 总结 ===")
    print("✅ 全部通过" if ok else "⚠️  存在需检查项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
