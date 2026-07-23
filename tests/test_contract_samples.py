"""
校验 schemas/contracts/samples 下 JSON 与 v1 契约一致，防止并行开发时样例漂移。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.contracts import (
    EmotionAnalyzeRequest,
    EmotionTags,
    InterventionRequest,
    InterventionResult,
    PipelineInput,
    PipelineOutput,
    RouteDecision,
    RouteRequest,
    SafetyCheckRequest,
    SafetyCheckResult,
)

_SAMPLES = Path(__file__).resolve().parent.parent / "schemas" / "contracts" / "samples"


def _load(name: str) -> dict:
    return json.loads((_SAMPLES / name).read_text(encoding="utf-8"))


def test_sample_json_matches_contracts():
    SafetyCheckRequest.model_validate(_load("safety_request.json"))
    SafetyCheckResult.model_validate(_load("safety_response.json"))

    EmotionAnalyzeRequest.model_validate(_load("emotion_request.json"))
    EmotionTags.model_validate(_load("emotion_response.json"))

    RouteRequest.model_validate(_load("router_request.json"))
    RouteDecision.model_validate(_load("router_response.json"))

    InterventionRequest.model_validate(_load("intervention_request.json"))
    InterventionResult.model_validate(_load("intervention_response.json"))

    for name in (
        "intervention_case_comfort.json",
        "intervention_case_knowledge.json",
        "intervention_case_crisis.json",
        "intervention_case_minimal.json",
    ):
        InterventionRequest.model_validate(_load(name))

    PipelineInput.model_validate(_load("pipeline_request.json"))
    PipelineOutput.model_validate(_load("pipeline_response.json"))
