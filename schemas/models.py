from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    user_input: str = Field(..., description="用户输入")
    user_id: Optional[str] = Field(default=None, description="用户ID")
    stream: bool = Field(default=False, description="是否流式输出")
    enable_thought_chain: bool = Field(default=True, description="是否启用思维链")
    enable_emotion_analysis: bool = Field(default=True, description="是否启用情绪分析")


class EmotionAnalysisResponse(BaseModel):
    primary_emotion: str
    intensity: int
    emotion_cues: List[str]
    underlying_needs: List[str]
    cognitive_distortions: List[str]
    safety_concerns: str


class ThoughtChainResponse(BaseModel):
    emotion_recognition: str
    emotion_intensity: int
    user_needs: List[str]
    therapy_approach: str
    reasoning_process: str
    response_strategy: str
    empathy_expression: str
    safety_check: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    thought_chain: Optional[ThoughtChainResponse] = None
    emotion_analysis: Optional[EmotionAnalysisResponse] = None
    suggested_techniques: List[str] = []
    safety_alert: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)


class SessionInfo(BaseModel):
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_active: datetime
    message_count: int
    emotion_history: List[Dict[str, Any]]
    key_topics: List[str]


class SessionListResponse(BaseModel):
    sessions: List[str]
    count: int


class EmotionTrendResponse(BaseModel):
    trend: str
    average_intensity: float
    recent_emotions: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_name: str
