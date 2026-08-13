"""演示：不启动服务，直接驱动主动式 SCID 访谈引擎走完一轮抑郁评估。

用法：
    & "D:\Anaconda\envs\emotion\python.exe" scripts/demo_scid_interview.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.assessment.scid_interview import SCIDInterviewEngine


def main() -> None:
    engine = SCIDInterviewEngine()

    # (用户输入, 被动 tracker 已匹配的判据) —— 模拟真实对话，按顺序喂
    turns = [
        ("医生，我最近心情不好，晚上也睡不好", ["sleep"]),      # 触发启动 → 问 gate
        ("对，几乎每天都这样，快一个月了", []),                   # gate 确认 → A1/A2 + 跳过已被动的 A3？没有 → 问 A3
        ("吃不下，瘦了好几斤", []),                              # A3 确认 → 问 A4
        ("没有，睡得还行", []),                                 # A4 否认 → 问 A5
        ("倒没觉得，就是很累", []),                              # A5 否认 → 问 A6
        ("嗯，很累，提不起劲", []),                              # A6 确认 → 问 A7
        ("觉得自己很没用，拖累了家里", []),                       # A7 确认 → 问 A8
        ("看书都看不进去了", []),                                # A8 确认 → 问 A9
        ("有时候觉得活着没意思", []),                            # A9 确认 → 双相筛查（安全链路接管）
        ("没有过那种特别兴奋的情况", []),                        # 双相否认 → 问功能损害
        ("有影响，上班都没心思", []),                             # 功能损害确认 → 出结论
    ]

    state = None
    print("=" * 70)
    for i, (user_text, passive) in enumerate(turns, 1):
        state, directive = engine.step_turn(state, user_text, passive)
        print(f"\n[第 {i} 轮] 用户说：{user_text}")
        if directive:
            print("→ 引擎指令（注入 LLM prompt）：")
            for line in directive.splitlines():
                print(f"   {line}")
        else:
            print("→ 无指令（访谈未进行/已结束）")
    print("\n" + "=" * 70)
    print("最终访谈状态：")
    print(f"  已确认：{state['criteria_confirmed'] if state else '-'}")
    print(f"  已排除：{state['criteria_denied'] if state else '-'}")
    print(f"  状态：{state['status'] if state else '-'}")


if __name__ == "__main__":
    main()
