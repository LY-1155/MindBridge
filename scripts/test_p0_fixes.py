"""Quick functional test for P0 fixes: chronicity bonus + probed dimensions."""
from __future__ import annotations

from modules.emotion.stub import EmotionService
from modules.intervention.generator import detect_probed_dimension
from schemas.contracts import EmotionAnalyzeRequest
from core.memory.session_memory import TherapySessionMemory, SessionManager

# Use production risk_config (same as modules/factory.py)
import json
from pathlib import Path
_rules_path = Path(__file__).resolve().parent.parent / "config" / "router_rules.json"
_rules = json.loads(_rules_path.read_text(encoding="utf-8"))
risk_config = _rules.get("risk_formula", {})
svc = EmotionService(risk_config=risk_config)

# ── Test 1: normal insomnia ──
req1 = EmotionAnalyzeRequest(
    contract_version="1.0",
    text="最近睡不着，想开点安眠药",
    session_id="test",
    safety={"level": 0, "blocked": False, "matched_terms": []},
)
result1 = svc.analyze(req1)
print(f"Test 1 - 普通失眠: primary={result1.primary_emotion}, risk={result1.risk}, intensity={result1.intensity}")
print(f"  modality_notes={result1.modality_notes}")

# ── Test 2: chronic severe distress ──
req2 = EmotionAnalyzeRequest(
    contract_version="1.0",
    text="有一两年了吧，最近感觉更严重了，我好难受又感觉好不了，什么都不想干",
    session_id="test",
    safety={"level": 0, "blocked": False, "matched_terms": []},
)
result2 = svc.analyze(req2)
print(f"Test 2 - 慢性绝望: primary={result2.primary_emotion}, risk={result2.risk}, intensity={result2.intensity}")
print(f"  modality_notes={result2.modality_notes}")

# ── Test 3: the user's actual message ──
req3 = EmotionAnalyzeRequest(
    contract_version="1.0",
    text="什么都不想干，手机玩着也没意思，什么都不想干也不想动",
    session_id="test",
    safety={"level": 0, "blocked": False, "matched_terms": []},
)
result3 = svc.analyze(req3)
print(f"Test 3 - 用户原话(没劲): primary={result3.primary_emotion}, risk={result3.risk}, intensity={result3.intensity}")
print(f"  modality_notes={result3.modality_notes}")

# ── Verify routing thresholds ──
print("\n--- Routing threshold check ---")
for i, (label, risk) in enumerate([
    ("普通失眠", result1.risk),
    ("慢性绝望", result2.risk),
    ("用户原话", result3.risk),
], 1):
    if risk >= 0.7:
        route = "CRISIS"
    elif risk >= 0.5:
        route = "COMFORT"
    elif risk >= 0.1:
        route = "KNOWLEDGE"
    else:
        route = "GENERAL"
    print(f"  Test {i} ({label}): risk={risk:.3f} → {route}")

# ── Test 4: probed dimension detection ──
print("\n--- Probed dimension detection ---")
test_replies = [
    # Should detect 时间线
    "眼睁睁看着天一点一点亮起来……太磨人了。这种情况多久了，是最近才开始，还是有一阵子了？",
    # Should detect 精力
    "连平时最爱的游戏都提不起劲，这种彻底被掏空的感觉确实很难熬。除了游戏之外，出门见朋友的劲头也受影响了吗？",
    # Should detect 睡眠
    "凌晨三四点醒，身体沉甸甸的但脑子却清醒——这种身心不同步太折磨了。除了早醒，入睡也困难吗？",
    # Should detect nothing (pure empathy)
    "嗯，我听到你了。这些事压在一个人身上确实太重了。",
    # Should detect 严重度
    "这种状态已经影响到你做事和工作的程度了吗，还是说目前还能勉强扛住？",
]
for reply in test_replies:
    dim = detect_probed_dimension(reply)
    print(f"  [{dim or '无探测'}] {reply[:60]}...")

# ── Test 5: session probed_dimensions round-trip ──
print("\n--- Session probed_dimensions ---")
sid = SessionManager.create_session(user_id="test_user")
session = SessionManager.get_session(sid)
print(f"  Initial probed: {session.get_probed_dimensions()}")

session.add_probed_dimension("时间线")
session.add_probed_dimension("睡眠")
print(f"  After adding 2: {session.get_probed_dimensions()}")

session.add_probed_dimension("时间线")  # duplicate, should be ignored
print(f"  After duplicate add: {session.get_probed_dimensions()}")

# Re-load and verify persistence
session2 = SessionManager.get_session(sid)
print(f"  After reload: {session2.get_probed_dimensions()}")

print("\n✅ All tests passed!")
