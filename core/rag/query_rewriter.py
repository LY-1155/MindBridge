"""Query Rewriter：检索前查询改写，提升心理知识库召回率

基于 Query Rewriting 技术（DeepSeek RAG Layer 1 / PreQRAG / HyDE 范式），
将用户口语化咨询问题改写为富含专业术语和同义词的检索关键词串。

设计原则：
- 不改原意，只补充术语和同义词
- 改写失败自动回退到原始查询
- 一次轻量 LLM 调用，延迟可控（< 1s）
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板 — 将口语改写为心理学关键词
# ---------------------------------------------------------------------------
# 每条示例覆盖一个心理学知识域，引导 LLM 按领域补全术语。
# 设计要点：
#   1. 不做分类 — 不限定单一类别，免得分错过滤掉正确答案
#   2. 不编造 — 只补术语和同义词，不生成原文没有的含义
#   3. 不覆盖 — 改写结果与原文拼接使用，不是替代原文
# ---------------------------------------------------------------------------
_REWRITE_SYSTEM_PROMPT = """你是一位心理咨询术语专家。将用户的日常表达改写成心理学知识库检索用关键词串。

## 规则
1. 保留原意，补充心理学专业术语和同义词
2. 不编造用户没说的症状或情绪
3. 只输出关键词，不要完整句子，不要解释
4. 将用户表述中的具体场景词（如"迟到"、"打翻水杯"）映射为背后的心理过程词（如"失败情境"、"失误"），但绝不将其改写为"时间管理"或"拖延"

## 示例
用户: 男朋友老躲着我发消息不回
关键词: 回避型依恋、亲密关系冲突、沟通模式、情感回避、关系焦虑

用户: 控制不住什么都往坏处想
关键词: 灾难化思维、认知扭曲、广泛性焦虑、CBT、情绪调节技巧

用户: 睡不着脑子里一直转停不下来
关键词: 入睡困难、思维反刍、失眠、睡眠卫生、放松训练、正念

用户: 做什么都没意思提不起劲
关键词: 快感缺失、兴趣减退、抑郁情绪、动力缺乏、行为激活

用户: 分手后一直走不出来
关键词: 延长哀伤障碍、丧失应对、依恋断裂、居丧支持、情感疗愈

用户: 在人多的场合就紧张出汗
关键词: 社交焦虑、回避行为、暴露疗法、自主神经反应、认知重构

用户: 总是担心自己得重病
关键词: 疑病症、健康焦虑、躯体症状障碍、疾病信念、CBT干预

用户: 每次迟到后都觉得自己一无是处
关键词: 失败情境、自我贬低、完美主义、认知扭曲、自尊受损、过度自责

用户: 经历过不好的事，想起来就心慌手抖冒冷汗
关键词: 创伤后应激、PTSD、侵入性回忆、过度警觉、创伤触发、躯体化反应、闪回

用户: 遇到烦心事不知道怎么调节自己，越憋越难受
关键词: 情绪调节困难、应对策略缺乏、压力管理、心理弹性、自我安抚、表达性宣泄、适应性应对"""

_REWRITE_USER_TEMPLATE = """用户: {query}
关键词:"""


class QueryRewriter:
    """查询改写器 — 检索前增强查询表达力

    用法::

        from core.llm.base import OpenAICompatibleAdapter
        llm = OpenAICompatibleAdapter()
        rewriter = QueryRewriter(llm)
        enhanced = rewriter.rewrite("男朋友老躲着我")
        # → "男朋友老躲着我 回避型依恋、亲密关系冲突、沟通模式"
    """

    def __init__(self, llm):
        """参数:
            llm: LangChain BaseChatModel 或 BaseLLMAdapter 实例
        """
        self._llm = llm

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def rewrite(self, query: str) -> str:
        """改写用户查询，返回拼接后的增强查询。

        原文 + 改写关键词，拼接传给检索器。
        改写失败时自动回退，返回原始查询。
        """
        if not query or not query.strip():
            return query

        try:
            keywords = self._call_llm(query.strip())
            if not keywords:
                logger.info("QueryRewriter 返回空，使用原始查询")
                return query

            # 拼接原文 + 关键词：BM25 命中原文，Chroma 命中语义扩展
            enhanced = f"{query} | {keywords}"
            logger.debug("QueryRewriter: %s → %s", query[:60], enhanced[:120])
            return enhanced

        except Exception:
            logger.warning("QueryRewriter 调用失败，回退原始查询", exc_info=True)
            return query

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _call_llm(self, query: str) -> str:
        """调用 LLM 生成关键词"""
        messages = [
            SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=_REWRITE_USER_TEMPLATE.format(query=query)),
        ]

        # 兼容两种 LLM 接口:
        #   BaseLLMAdapter.invoke(messages) → AIMessage
        #   BaseChatModel.invoke(messages)   → AIMessage
        response = self._llm.invoke(messages)

        # 提取文本内容
        if hasattr(response, "content"):
            raw = response.content
        else:
            raw = str(response)

        return self._clean(raw)

    @staticmethod
    def _clean(raw: str) -> str:
        """清理 LLM 输出：去空白、去多余的标点"""
        text = raw.strip()
        # 去掉 LLM 偶尔输出的前缀标签
        for prefix in ("关键词:", "关键词：", "Keywords:", "标签:", "标签："):
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        return text
