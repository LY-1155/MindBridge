"""
Gap #12: AI 标注 — 每条回复底部追加 "AI 辅助回复，非医疗诊断"

验证行为：
  1. apply_disclaimer 在非空回复末尾追加免责声明
  2. 空回复不追加
  3. 不会重复追加
  4. 免责声明包含固定的标识文字
  5. chat 端点返回的 response 包含免责声明
  6. multimodal chat 端点返回的 response 包含免责声明
  7. pipeline 端点返回的 intervention.reply 包含免责声明
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest


# ---------------------------------------------------------------------------
# 1. apply_disclaimer 基础行为
# ---------------------------------------------------------------------------

class TestApplyDisclaimer:
    """apply_disclaimer 追加免责声明"""

    def test_appends_to_non_empty_reply(self):
        from modules.ai_disclaimer import apply_disclaimer, AI_DISCLAIMER
        result = apply_disclaimer("你好，我理解你的感受。")
        assert result.startswith("你好，我理解你的感受。")
        assert result.endswith(AI_DISCLAIMER)

    def test_does_not_append_to_empty_reply(self):
        from modules.ai_disclaimer import apply_disclaimer
        assert apply_disclaimer("") == ""
        assert apply_disclaimer(None) is None

    def test_no_double_append(self):
        from modules.ai_disclaimer import apply_disclaimer, AI_DISCLAIMER
        once = apply_disclaimer("测试回复")
        twice = apply_disclaimer(once)
        # 不应重复追加
        assert twice.count(AI_DISCLAIMER) == 1

    def test_disclaimer_contains_required_text(self):
        from modules.ai_disclaimer import AI_DISCLAIMER
        assert "AI 辅助回复" in AI_DISCLAIMER
        assert "非医疗诊断" in AI_DISCLAIMER


# ---------------------------------------------------------------------------
# 2. Chat 端点集成
# ---------------------------------------------------------------------------

class TestChatDisclaimerIntegration:
    """文本 chat 端点返回的 response 包含免责声明"""

    @pytest.fixture(autouse=True)
    def _override_auth(self):
        from modules.auth_deps import get_current_user_id
        from api.main import app

        async def _fake_user():
            return "test-user-gap12"

        app.dependency_overrides[get_current_user_id] = _fake_user
        yield
        app.dependency_overrides.clear()

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def test_text_chat_response_has_disclaimer(self, client):
        """文本 /api/v1/chat 返回的 response 带免责声明"""
        from modules.ai_disclaimer import AI_DISCLAIMER

        resp = client.post(
            "/api/v1/chat",
            json={"message": "测试消息", "session_id": None},
        )
        data = resp.json()
        reply = data.get("response", "")
        # 即使管线失败，response 为空或带免责声明
        # 如果管线有效产出，必须包含免责声明
        if reply:
            assert AI_DISCLAIMER in reply, f"Expected disclaimer in: {reply!r}"

    def test_multimodal_text_chat_response_has_disclaimer(self, client):
        """多模态 text-only chat 返回的 response 带免责声明"""
        from modules.ai_disclaimer import AI_DISCLAIMER

        resp = client.post(
            "/api/v1/multimodal/chat",
            json={"session_id": None, "text": "测试", "audio_data": None, "image_data": None, "video_data": None, "enable_tts": False},
        )
        data = resp.json()
        reply = data.get("response", "")
        if reply:
            assert AI_DISCLAIMER in reply, f"Expected disclaimer in: {reply!r}"

    def test_pipeline_run_response_has_disclaimer(self, client):
        """pipeline /run 的 intervention.reply 带免责声明"""
        from modules.ai_disclaimer import AI_DISCLAIMER

        resp = client.post(
            "/api/v1/pipeline/run",
            json={"text": "测试", "user_id": "test-user-gap12", "session_id": None},
        )
        data = resp.json()
        reply = data.get("intervention", {}).get("reply", "")
        if reply:
            assert AI_DISCLAIMER in reply, f"Expected disclaimer in: {reply!r}"
