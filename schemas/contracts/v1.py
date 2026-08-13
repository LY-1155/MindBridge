"""
模块间 JSON 契约（v1）
====================

四模块并行开发时，仅以本文件中的模型作为跨模块边界；字段变更请升级 contract_version 并评审。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


CONTRACT_VERSION = "1.4"

SUPPORTED_EMOTIONS = Literal[
    "neutral", "anxiety", "sadness", "anger",
    "fear", "stress", "happiness", "confusion",
    "distress",  # 安全短路时的占位情绪
]


class SafetyCheckRequest(BaseModel):
    """输入与安全过滤模块 · 输入"""

    contract_version: str = Field(default=CONTRACT_VERSION, description="契约版本")
    text: str = Field(..., description="待检测文本（可由上游 ASR 合并）")
    session_id: Optional[str] = None
    locale: str = Field(default="zh-CN", description="语言区域")


class SafetyCheckResult(BaseModel):
    """输入与安全过滤模块 · 输出"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    level: int = Field(
        default=0,
        description="分级：0 通过；1 记录；2 紧急（与业务词表对齐后可调整含义）",
    )
    blocked: bool = Field(default=False, description="为 True 时流水线可短路至危机干预")
    matched_terms: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict, description="扩展字段，便于调试与审计")


class EmotionAnalyzeRequest(BaseModel):
    """情感分析模块 · 输入（接收安全模块输出 JSON）"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    text: str
    audio_path: Optional[str] = Field(
        default=None,
        description="可选，原始音频文件路径；用于语音情绪分析",
    )
    safety: Dict[str, Any] = Field(
        ...,
        description="SafetyCheckResult.model_dump()，模块间仅用 JSON 传递",
    )
    pre_extracted_audio_emotion: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选，API 层合并 SenseVoice 调用后的预提取情绪，跳过重复推理",
    )
    pre_extracted_visual_emotion: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选，VideoPreprocessor 前置处理后的视觉情绪信号（多帧聚合产物）",
    )
    session_id: Optional[str] = None


class EmotionTags(BaseModel):
    """情感分析模块 · 输出"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    primary_emotion: SUPPORTED_EMOTIONS = Field(default="neutral", description="主情绪标签")
    intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="强度 0~1")
    risk: float = Field(default=0.0, ge=0.0, le=1.0, description="风险 0~1")
    modality_notes: Dict[str, Any] = Field(
        default_factory=dict,
        description="音视频等模态附加信息",
    )
    intent: Optional[str] = Field(
        default=None,
        description="文本意图：information（信息提问）、emotion_expression（情感表达）、casual_chat（闲聊）、unknown",
    )


class RouteRequest(BaseModel):
    """智能路由模块 · 输入"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    emotion: Dict[str, Any] = Field(..., description="EmotionTags.model_dump()")
    safety: Dict[str, Any] = Field(..., description="SafetyCheckResult.model_dump()")


class RouteDecision(BaseModel):
    """智能路由模块 · 输出"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    route: Literal["general", "comfort", "knowledge", "crisis"] = "general"
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    meta: Dict[str, Any] = Field(default_factory=dict, description="路由决策元数据：风险段位、修正因子、信号冲突等")


class InterventionRequest(BaseModel):
    """干预闭环模块 · 输入（由 pipeline 编排组装，或单测直接 POST body）。"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    user_text: str
    user_id: Optional[str] = None
    route: Dict[str, Any] = Field(..., description="RouteDecision.model_dump()")
    emotion: Dict[str, Any] = Field(..., description="EmotionTags.model_dump()")
    safety: Dict[str, Any] = Field(..., description="SafetyCheckResult.model_dump()")
    session_id: Optional[str] = None
    safety_verdict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="语义安全评估器裁决 dict（SafetyVerdict.to_dict()）；None=未运行/无锚点。医生模式下驱动 crisis/probe/no_risk 三分支。",
    )


class InterventionResult(BaseModel):
    """干预闭环模块 · 输出（API JSON 即此模型；PipelineOutput.intervention 为 model_dump）。"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    reply: str = Field(default="", description="对用户主回复")
    empathy: str = ""
    suggestion: str = ""
    action_items: List[str] = Field(default_factory=list)
    chain_of_thought: Optional[str] = Field(
        default=None,
        description="可选，对内展示推理轨迹",
    )
    emergency_triggered: bool = False
    meta: Dict[str, Any] = Field(default_factory=dict)


class PipelineInput(BaseModel):
    """端到端流水线 · 输入"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    text: str
    user_id: Optional[str] = Field(default=None, description="发起请求的用户 ID（用于安全标记归属 + 会话绑定）")
    audio_path: Optional[str] = Field(
        default=None,
        description="可选，原始音频文件路径；用于语音情绪分析",
    )
    pre_extracted_audio_emotion: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选，API 层合并 SenseVoice 调用后的预提取情绪",
    )
    session_id: Optional[str] = None


class PipelineOutput(BaseModel):
    """端到端流水线 · 输出（各阶段均为 JSON 可序列化 dict）"""

    contract_version: str = Field(default=CONTRACT_VERSION)
    safety: Dict[str, Any]
    emotion: Dict[str, Any]
    route: Dict[str, Any]
    intervention: Dict[str, Any]
    stopped_after_safety: bool = Field(
        default=False,
        description="安全短路时后续阶段可为占位数据",
    )
    safety_verdict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="语义安全评估器裁决 dict；None=未运行/无锚点",
    )
