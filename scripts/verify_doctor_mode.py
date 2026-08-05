"""验证 DOCTOR_MODE 家庭模式对话（真实 LLM）。

用法：& "D:\Anaconda\envs\emotion\python.exe" scripts\verify_doctor_mode.py
跑一段 4 轮青少年家庭场景对话，打印回复 + session state。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from schemas.contracts import PipelineInput
from pipeline.orchestrator import run_pipeline

assert settings.DOCTOR_MODE, "DOCTOR_MODE 未开启！请在 .env 设置 DOCTOR_MODE=true"
print(f"== DOCTOR_MODE={settings.DOCTOR_MODE} persona={settings.DOCTOR_PERSONA} model={settings.MODEL_NAME} ==\n")

# 一段青少年家庭场景对话（4 轮）
CONVERSATION = [
    "孩子最近不上学了，天天躲在房间打游戏，怎么说都不听",
    "他爸脾气大，一说孩子就吼，两个人就吵起来，我也很烦",
    "每次我们吵完，孩子第二天就更不愿意去上学了",
    "我也试过好好跟他讲道理，可他就是不听，我实在不知道怎么办了",
]


def show_session(session_id):
    try:
        from core.memory.session_memory import SessionManager
        session = SessionManager.get_session(session_id)
        meta = session.metadata
        print(f"  [session] phase={meta.phase} hypothesis={meta.working_hypothesis!r}")
        print(f"  [session] family={[m['role'] for m in meta.family_members]}")
        print(f"  [session] scid_flags={ {k: v['count'] for k, v in meta.scid_flags.items()} }")
    except Exception as e:
        print(f"  [session] 读取失败: {e}")


for i, msg in enumerate(CONVERSATION):
    print(f"── 第 {i+1} 轮 ──")
    print(f"用户: {msg}")
    inp = PipelineInput(
        text=msg,
        user_id="verify-user",
        session_id="verify-family-session",
    )
    try:
        out = run_pipeline(inp)
    except Exception as e:
        print(f"  [ERROR] 管线失败: {e}")
        continue

    route = out.route.get("route", "?")
    emotion = out.emotion.get("primary_emotion", "?")
    risk = out.emotion.get("risk", 0)
    reply = out.intervention.get("reply", "")
    impl = out.intervention.get("meta", {}).get("implementation", "?")
    emergency = out.intervention.get("emergency_triggered", False)

    print(f"  route={route} emotion={emotion} risk={risk:.2f} impl={impl} emergency={emergency}")
    print(f"  AI: {reply}\n")
    show_session("verify-family-session")
    print()
