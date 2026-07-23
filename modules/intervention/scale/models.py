"""量表数据模型：ScaleItem / ScaleThreshold / ScaleConfig"""
from __future__ import annotations

import json
import os
from typing import List, Dict, Optional
from pydantic import BaseModel


SCALES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "knowledge", "private", "scales"
)


class ScaleItem(BaseModel):
    index: int
    dimension: str = ""
    concept: str
    anchors: Dict[str, str] = {}
    # LSAS 专用：情境描述和分类（performance / social）
    situation: Optional[str] = None
    category: Optional[str] = None


class ScaleThreshold(BaseModel):
    min: int
    max: int
    level: str
    label: str


class EscalationRule(BaseModel):
    total_score_threshold: int
    item_triggers: List[int]
    item_threshold: int


class ScaleConfig(BaseModel):
    name: str
    display_name: str
    description: str
    items: List[ScaleItem]
    thresholds: List[ScaleThreshold]
    escalation: EscalationRule
    # 计分模式：sum(求和) | count_yes(0/1计数) | asrs(非对称阈值) | lsas(双轴) | mdq(组合逻辑)
    scoring_mode: str = "sum"
    # 非标准模式专用配置，可选
    positive_rule: Optional[Dict] = None      # mdq: {"q1_min": 7, "q2_required": true, "q3_min": 2}
    item_thresholds: Optional[Dict[str, int]] = None  # asrs: {"0": 2, ...} 每题达标分
    positive_count: Optional[int] = None      # asrs / count_yes: 阳性所需达标题数
    dual_dimensions: Optional[Dict] = None    # lsas: {"fear": {...}, "avoidance": {...}}

    @classmethod
    def from_json(cls, scale_name: str) -> "ScaleConfig":
        # normalize: lowercase + strip dashes for filesystem
        filename = scale_name.lower().replace("-", "")
        path = os.path.join(SCALES_DIR, f"{filename}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class ScaleState(BaseModel):
    """会话中量表施测的实时状态，存于 SessionMetadata.scale_state。"""
    scale_name: str
    current_item_index: int = 0
    scores: List[int] = []
    status: str = "in_progress"  # in_progress | completed | abandoned
    wander_count: int = 0
    total_score: Optional[int] = None
    level: Optional[str] = None
    escalation_flag: bool = False
    # LSAS 双轴：当前正在评哪个维度（"fear" / "avoidance"），其他量表为 None
    current_dimension: Optional[str] = None
    # D10: 串行执行队列，共病场景存后续量表名列表
    pending_scales: List[str] = []


class ScaleTurnResult(BaseModel):
    """process_turn 的返回结果。"""
    reply: str
    is_complete: bool
    total_score: Optional[int] = None
    level: Optional[str] = None
    escalation_flag: bool = False
