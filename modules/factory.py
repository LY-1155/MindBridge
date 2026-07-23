"""
按环境变量装配四模块实现（Mock vs Stub）。
=======================================

并行开发：`MOCK_*=true` 时使用 Mock；为 `false` 时对应该模块 Stub，
团队可在 Stub 内替换为真实实现或改为从新模块 import。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from config.settings import Settings
from modules.emotion.mock import MockEmotionService

logger = logging.getLogger(__name__)
from modules.emotion.stub import EmotionService
from modules.intervention.mock import MockInterventionService
from modules.intervention.service import InterventionService
from modules.router.mock import MockRouterService
from modules.router.router_service import RouterService
from modules.router.stub import StubRouterService
from modules.safety.mock import MockSafetyService
from modules.safety.stub import StubSafetyService
from modules.ports import EmotionPort, InterventionPort, RouterPort, SafetyPort

_RULES_PATH = str(Path(__file__).resolve().parent.parent / "config" / "router_rules.json")


def _load_rules() -> Dict[str, Any]:
    return json.loads(Path(_RULES_PATH).read_text(encoding="utf-8"))


@dataclass
class PipelineServices:
    safety: SafetyPort
    emotion: EmotionPort
    router: RouterPort
    intervention: InterventionPort


def get_safety_service(settings: Settings) -> SafetyPort:
    return MockSafetyService() if settings.MOCK_SAFETY else StubSafetyService()


def get_emotion_service(settings: Settings) -> EmotionPort:
    if settings.MOCK_EMOTION:
        return MockEmotionService()

    # 根据 EMOTION_ENGINE 选择文本情绪引擎
    from modules.emotion.keyword_engine import KeywordEmotionEngine

    engine = KeywordEmotionEngine()

    if settings.EMOTION_ENGINE == "onnx":
        try:
            from modules.emotion.onnx_engine import ONNXEmotionEngine

            tokenizer_path = settings.EMOTION_ONNX_TOKENIZER_PATH or ""
            onnx_engine = ONNXEmotionEngine(
                model_path=settings.EMOTION_ONNX_MODEL_PATH,
                tokenizer_path=tokenizer_path or "",
            )
            if onnx_engine.is_ready:
                engine = onnx_engine
            else:
                logger.warning("[FACTORY] ONNX engine not ready, falling back to keyword")
        except Exception as e:
            logger.warning("[FACTORY] ONNX engine failed to load: %s, falling back to keyword", e)

    rules = _load_rules()
    risk_config = rules.get("risk_formula", {})
    return EmotionService(text_engine=engine, risk_config=risk_config)


def get_router_service(settings: Settings) -> RouterPort:
    return MockRouterService() if settings.MOCK_ROUTER else RouterService(_RULES_PATH)


def get_intervention_service(settings: Settings) -> InterventionPort:
    # 干预闭环：MOCK_INTERVENTION=true 用 Mock，false 用 InterventionService（crisis/comfort/knowledge 三路分支）
    return MockInterventionService() if settings.MOCK_INTERVENTION else InterventionService()


def build_pipeline_services(settings: Settings) -> PipelineServices:
    return PipelineServices(
        safety=get_safety_service(settings),
        emotion=get_emotion_service(settings),
        router=get_router_service(settings),
        intervention=get_intervention_service(settings),
    )
