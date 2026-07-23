"""ScaleScorer：独立 LLM 计分，对照锚点判定 0-3 或 -1 无效。
使用轻量模型，与对话生成的模型分离。模型通过 .env 中 SCORING_MODEL_NAME 配置。
LLM 输出经 PydanticOutputParser 结构化解析，替代正则提取。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from modules.intervention.scale.models import ScaleItem
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX


class ScaleScoreResult(BaseModel):
    """LLM 计分结构化输出。"""
    score: int = Field(description="锚点分数，0 到 N 表示严重程度递增，-1 表示回复与量表维度完全无关")


SCORING_SYSTEM_PROMPT = """你是一个严格的心理量表计分器。你的任务是阅读用户的自然语言回复，判断其在指定维度上的严重程度。

## 评分的维度
- 维度名：{dimension}
- 症状范围：{concept}

## 评分标准（锚点）
{anchors_text}

## 无效情况
如果用户的回复与上述维度完全无关（用户在问知识问题、抱怨其他事情、或者话题完全不涉及自己的症状体验），返回 -1。

## 口语应答
用户可能用简短口语应答。请结合回复的语气、内容和上下文推断其对应频率/程度的锚点分数。不要仅因回答简短就判为 -1。

{format_instructions}"""


class ScaleScorer:
    """独立计分器，每个量表题目调用一次 LLM（默认轻量模型）。
    使用 PydanticOutputParser 做结构化输出解析。"""

    def __init__(self, llm=None, *, model_name: str = None):
        from config.settings import settings
        _model = model_name or settings.SCORING_MODEL_NAME
        if llm is not None:
            self._llm = llm
            self._model_name = _model
        else:
            from core.llm.base import get_llm_adapter, LLMConfig
            self._model_name = _model
            self._llm = get_llm_adapter("openai_compatible", config=LLMConfig(model_name=self._model_name))

    def score(self, user_reply: str, item: ScaleItem) -> int:
        """单维计分：LLM 结构化输出，极短回复做一次复盘重试。"""
        anchors_text = self._anchors_text(item.anchors)
        prompt = SCORING_SYSTEM_PROMPT.format(
            dimension=item.dimension or item.situation or "",
            concept=item.concept,
            anchors_text=anchors_text,
            format_instructions=self._format_instructions(),
        )
        result = self._invoke_score(prompt, user_reply)
        result = self._retry_if_suspicious(prompt, user_reply, result)
        return result

    def score_dual(self, user_reply: str, item: ScaleItem, dim_key: str, dim_config: dict) -> int:
        """LSAS 双轴计分：对指定维度（fear / avoidance）独立计分。"""
        anchors_text = self._anchors_text(dim_config.get("anchors", {}))
        dim_label = dim_config.get("label", dim_key)
        prompt = SCORING_SYSTEM_PROMPT.format(
            dimension=f"{item.situation or item.concept} — {dim_label}",
            concept=f"在「{item.situation or item.concept}」这个情境下，{dim_label}的程度",
            anchors_text=anchors_text,
            format_instructions=self._format_instructions(),
        )
        result = self._invoke_score(prompt, user_reply)
        result = self._retry_if_suspicious(prompt, user_reply, result)
        return result

    # ── internal ────────────────────────────────────────────

    @staticmethod
    def _format_instructions() -> str:
        parser = PydanticOutputParser(pydantic_object=ScaleScoreResult)
        return parser.get_format_instructions()

    @staticmethod
    def _anchors_text(anchors: dict) -> str:
        """将锚点字典转为 prompt 文本。"""
        lines = []
        for k in sorted(anchors.keys(), key=lambda x: int(x)):
            lines.append(f"- {k} 分：{anchors[k]}")
        return "\n".join(lines)

    def _retry_if_suspicious(self, base_prompt: str, user_reply: str, first_result: int) -> int:
        """极短回复（≤5 字符）且首轮打分可疑时，给 LLM 一次复盘机会。"""
        text = user_reply.strip()
        if len(text) > 5:
            return first_result
        if first_result not in (0, -1):
            return first_result

        retry_hint = (
            "\n\n## 重要提醒\n"
            "用户刚刚给出的回复（\"" + text + "\"）非常简短，但这是在认真回答量表问题，"
            "不是语气词。注意：\n"
            "- 如果回复表达了肯定/确认（如\"会\"、\"有\"、\"嗯\"、\"是的\"），"
            "说明用户确认了症状存在。虽然没写明频率，但确认症状存在本身就排除了 0 分（完全否定）。"
            "请据此重新给分。\n"
            "- 如果回复表达了否认（如\"不\"、\"不会\"），才应给最低分。"
        )
        return self._invoke_score(base_prompt + retry_hint, user_reply)

    def _invoke_score(self, prompt: str, user_reply: str) -> int:
        parser = PydanticOutputParser(pydantic_object=ScaleScoreResult)

        prompt = prompt + INSTRUCTION_HIERARCHY_SUFFIX
        wrapped = wrap_user_text(user_reply)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"用户的回复：\n{wrapped}"),
        ]
        response = self._llm.invoke(messages)
        text = (response.content if hasattr(response, "content") else str(response)).strip()

        try:
            parsed = parser.parse(text)
            return parsed.score
        except Exception:
            return -1
