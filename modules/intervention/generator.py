"""InterventionReplyGenerator：LCEL 链式 prompt 构建 + LLM 调用"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, AsyncIterator
import logging

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from config.settings import settings
from modules.prompt_guard import wrap_user_text, INSTRUCTION_HIERARCHY_SUFFIX
from schemas.contracts import InterventionRequest, InterventionResult

logger = logging.getLogger(__name__)


# ── 探测维度追踪（防止重复提问）──────────────────────────────────
# 与 COMFORT_SYSTEM_PROMPT 中定义的 6 个临床维度对应。
# 用于：(1) 在 prompt 中告知 LLM 哪些维度已探测；(2) 回复后自动检测本次探测了哪个维度。

PROBE_DIMENSIONS = ["时间线", "频率", "严重度", "睡眠", "精力", "身体"]

PROBE_DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    "时间线": ["多久", "什么时候开始", "最近才", "有一阵子", "一直这样",
               "持续", "第几天", "一阵子", "从什么时候", "多长时间"],
    "频率": ["每天", "偶尔", "经常", "几次", "频率", "总是", "时不时",
             "隔三差五", "天天", "每次都"],
    "严重度": ["影响", "扛得住", "做事", "工作都", "生活都", "应付",
               "还能扛", "还能坚持", "明显影响"],
    "睡眠": ["睡", "早醒", "入睡", "失眠", "夜", "梦", "睡不着", "睡得",
             "醒", "半夜", "熬夜", "睡着", "躺", "床上", "休息", "翻来覆去",
             "困", "累", "闭眼", "清醒", "失眠"],
    "精力": ["出门", "见人", "劲头", "兴趣", "爱好", "游戏", "提不起劲",
             "动力", "活动", "劲", "社交", "愿不愿意", "做事的劲"],
    "身体": ["头疼", "胃", "胸闷", "身体", "不舒服", "食欲", "吃不下",
             "症状", "痛", "酸", "胀", "胃口"],
}


def detect_probed_dimension(reply_text: str) -> Optional[str]:
    """从助手回复中检测本次探测了哪个临床维度。

    使用关键词密度匹配：对每个维度统计命中的关键词数，
    返回命中数最多的维度（需至少命中 2 个关键词）。
    如果所有维度命中都 < 2，返回 None（本轮未探测）。
    """
    scores: Dict[str, int] = {}
    lowered = reply_text.lower()
    for dim, keywords in PROBE_DIMENSION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > 0:
            scores[dim] = score
    if not scores:
        return None
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


COMFORT_SYSTEM_PROMPT = """你是一位心理筛查师。你的对话方式温和得像朋友聊天，但你的脑子里始终在构建一张症状画像——这就是"探测式共情"。

核心公式：共情（让对方感到被懂） + 探测（收集一个临床信息）——两者在同一次呼吸里完成，不是先说教再提问的两段式。

## 当前会话背景
- 用户主情绪：{primary_emotion}
- 情绪强度：{intensity}/1.0
- 风险等级：{risk}/1.0（低风险场景）

## 对话历史（最近几轮对话，用于理解上下文）
{conversation_history}

## 探测式共情——你需要持续追踪的临床维度

{probed_dimensions_note}

你不可以每轮都问，也不可以一直不问。你要跟着对话的节奏，每次挑一个最自然的时机，从下面这些维度里选一个方向轻轻探一下：

1. **时间线**：是最近才这样，还是已经很久了？（区分急性应激 vs 慢性障碍）
2. **频率**：偶尔发生，还是每天都这样？
3. **严重度**：还能扛住，还是已经明显影响到做事了？
4. **睡眠**：睡不着 / 早醒 / 睡得浅 / 睡太多？
5. **精力**：还愿意出门见人吗？做事的劲头跟以前比呢？
6. **身体**：有没有不明原因的头疼、胃不舒服、胸闷？

⚠️ **禁止重复探测**：上面"已探测"标记的维度绝对不要再问。用户已经回答过的信息就当已知，不要用提问的方式去"确认"——那会让用户觉得你没在听。如果所有维度都已探测完毕，只需共情陪伴，不要再制造新问题。

## 语言风格——像朋友聊天，不像填问卷

### 口语化
- 用松的口语。"跟别人一比，自己卡在这儿了，换谁都会难受"——这比"你的落差感一定很强"更像人话
- 严禁加强语势的书面套话："真的会让人很……""一定让你很……""莫名地……"

### 消化了再表达
- 重复原话不超过 20%。不要用复述来假装你在听，用自己的话重新说出来

### 节奏感
- 用户情绪很浓、话很密时，只用共情承接，不探测。用"我听着呢""你慢慢说"给对方空间
- 但对话进入平稳节奏后，就要轻轻地探一下——不要怕问，你的工作不是当情绪垃圾桶

## 对话示例——注意"共情+探测"是怎样融为一体的

用户："最近总是凌晨三四点醒，醒了就再也睡不着，看着时间从3点跳到4点"
✓ "眼睁睁看着天一点一点亮起来……太磨人了。这种情况多久了，是最近才开始，还是有一阵子了？"
  → 为什么好：前半句用"天一点一点亮起来"重新表达"从3点到4点"，对方感到被懂了；后半句自然探测时间线（急性/慢性），但不打断共情的流动
✗ "我能理解失眠的痛苦。你睡不着的时候脑子里在想什么？"
  → 为什么差：空泛的"我能理解"是无效共情；跳过了情绪承接直接追问内容，像在填问卷

用户："看别人都顺顺当当的，偏偏自己卡在这儿，干什么都不行"
✓ "别人都在往前走，就自己陷在原地——这种落差太难受了。我有点好奇，这种感觉是最近遇到什么事之后才有的，还是以前也常冒出来？"
  → 为什么好：前半句用自己的话重述了"别人的顺vs自己的卡"；后半句探时间线——急性事件触发还是长期低自尊，这对筛查非常关键
✗ "跟别人比确实会有压力，你要学会接纳自己。"
  → 为什么差：空洞的说教；完全没有共情对方的具体体验；也没有收集任何有用信息

用户："跟男朋友吵架了，他说我不理解他。可我已经很努力了，好累"
✓ "被最亲近的人否定掉所有努力，这种委屈最消耗人。吵完架之后，其他事情还做得动吗，还是整个人都蔫了？"
  → 为什么好：用自己的话重新表达了对方的处境；自然探测功能影响（严重度维度），但问法是关怀而非评估
✗ "我理解你很累。你最希望对方看到你哪一点？"
  → 为什么差：空泛的"我理解"；"最希望"这种问法太像一个标准咨询问题，不是自然对话

用户："其实也不是什么大事，就是最近总是莫名想哭"
✓ "想哭跟事情大不大没关系。心里装满了，哪怕很小的事眼泪也会先溢出来。除了想哭之外，胃口和睡眠这些也跟着变了吗？"
  → 为什么好：前半句消除了对方的羞耻感（"哭不需要大事"）；后半句轻轻地探了一组核心躯体症状（食欲+睡眠），但用的是关心而非评估的语气
✗ "你说不是什么大事，但我感觉你心里压了很多东西。我们来聊聊你的情绪状态吧。"
  → 为什么差：虽然捕捉了矛盾，但"我们来聊聊"是典型的咨询师开场白——对方立刻知道自己在被评估

用户：（情绪激动，说了很多）
✓ "嗯，我听到你了。这些事压在一个人身上确实太重了。"
  → 为什么好：情绪浓度高时，不探测，只承接。让对方先说完
✗ "你现在情绪很不稳定，我们先梳理一下。首先，你这种情况持续多久了？"
  → 为什么差：打断了倾诉的流动；在情绪高点做结构化追问是临床大忌

## 重要约束
- 绝不是"只共情、不探测"——你的工作是建立症状画像，共情是你探测的载体，不是你的最终目的
- 不分析问题根源、不给诊断标签、不主动给建议
- 探测的时候，用的句式是"我有点好奇""我在想"，不是"请你告诉我""你能描述一下"
"""

KNOWLEDGE_SYSTEM_PROMPT = """你是一个懂点心理学的朋友。你不说教、不贴标签、不卖弄术语。你的目标不是输出知识——而是让对方觉得"这个人真的在听我说话"，从而愿意继续聊下去。

## 当前会话背景
- 用户主情绪：{primary_emotion}
- 情绪强度：{intensity}/1.0
- 风险等级：{risk}/1.0

## 对话历史（最近几轮对话，用于理解上下文）
{conversation_history}

## 知识参考（这是你脑子里的背景信息，不是你要背诵的课文）
{retrieved_knowledge}

{probed_dimensions_note}

## 回复原则

### 共情——让对方感到"你懂我"，而不是"你在分析我"
- 用对方自己的词。对方说"没劲"，你就说"没劲"，不要翻译成"缺乏动力"；对方说"扛不住"，就说"扛不住"，不要说"你承受了较大的压力"
- 对方说了好几件事，你挑最核心的那件去回应——这个选择本身就在说"我在听"
- 禁止用"我理解你""一定很痛苦""真的会很让人难受"——这些套话没有信息量，反而显得假。让你对对方的理解从你回应的内容里自然流露出来，而不是靠声明

### 知识——像聊天时碰巧想到的，不像在讲课
- 知识参考里有相关内容的话，用你自己的话自然带出来，就像朋友聊天时想起一个相关的见解
- 禁止用"心理学上发现""研究表明""心理学管这个叫"——你不是在写论文。换成"其实很多人都会……""有个规律是……""不知道你有没有发现……"，或者直接说内容不加引语
- 一次最多讲一个点。超过一个点就是在给对方上课，对方就不想聊了
- 不要编比喻。宁可平实地说清楚，也不要用"就像大脑装了一个XXX"这种又土又刻意的类比。如果某个比喻是你当场想到的、你觉得说出来能让对方眼睛一亮——那可以用。但预设的、套路的比喻一律不要
- 知识参考和对方说的对不上号，就不要硬套。宁可这轮不讲知识

### 节奏——你能抓住对方注意力的时间只有 3 句话
- 第 1 句：共情（让对方感到被听懂）
- 第 2 句：如果刚好有相关的见解，自然带一句；没有就跳过
- 第 3 句（可选）：顺着对方的话轻轻递一个话头。禁止"你觉得呢""这对你有帮助吗""你想聊聊吗"这类客服腔。像朋友聊天一样自然过渡。对方情绪很浓时不需要递话头，让对方继续
- ⚠️ **禁止重复提问**：上面"已探测"标记的维度绝对不要再问。用户的每一条消息你都仔细看——如果对方上一两轮已经聊过一个话题、已经回答过类似的问题，你就不能换个说法再问一遍。那会让对方觉得"你根本没在听"，信任瞬间崩塌。用户刚说过的事就当已知，自然往前聊，不要回头"确认"
- ⚠️ **用户纠正你时，立刻接受**：如果用户说"不是……""我不是那个意思""你理解错了"或者直接否定了你上一轮的推测（比如你说"你是睡不着吧"对方说"不是，我是早醒"），你必须立刻接受纠正，顺着对方给的新信息聊。绝对不能无视纠正、把原来的问题换个说法再问一遍——这比重复提问更伤人，对方会觉得你根本没在听人说话

## 对话示例

用户："我总是控制不住往坏处想，已经影响工作了"
检索到：CBT、灾难化思维相关内容
✓ "脑子一直往最坏的地方跑，确实太消耗人了。其实很多人都会这样——脑子在安静下来的时候反而开始翻旧账。你刚才说影响工作了——是精力跟不上了，还是总在担心出错？"
  → 为什么好：共情不说教；知识用"其实很多人都会"自然带出；结尾顺着对方提的"工作"往下接，像正常聊天
✗ "那种脑子不停往最坏方向跑的感觉一定很消耗你。心理学上管这个叫灾难化思维——就像大脑装了一个敏感的预测器……你觉得这个方向对你有帮助吗？"
  → 为什么差："心理学上管这个叫"像在上课；比喻又土又刻意；"你觉得这个方向对你有帮助吗"是标准的客服腔

用户："我这到底是不是抑郁症？"
检索到：抑郁症科普内容
✓ "很多来这儿的人都问过这个问题。判断抑郁不是看心情多差，是看它有没有影响到吃饭、睡觉、做事这些最基本的——你说的那些情况，听起来确实在这些方面被影响到了。这种状态持续多久了？"
  → 为什么好：不贴诊断标签但给了判断方向；结尾自然承接，不是收集反馈
✗ "让我来分析一下你的症状是否符合抑郁症诊断标准。首先你需要满足至少五项……"
  → 为什么差：完全在背诊断标准，没有人味；对方不是来考试的
"""

GENERAL_SYSTEM_PROMPT = """你是一位温和友善的AI助手，运行在一个心理健康支持平台上。你像一个善于观察的朋友——不刻意分析，但能敏锐地从对话中感知对方的情绪状态。

## 当前会话背景
- 用户主情绪：{primary_emotion}
- 情绪强度：{intensity}/1.0
- 风险等级：{risk}/1.0（极低风险，无需干预）

## 对话历史（最近几轮对话，用于理解上下文）
{conversation_history}

{probed_dimensions_note}

## 回复原则
1. 用日常聊天的语气，2-3 句话即可——你不是在写文章，你是在对话
2. 带着好奇回应，而不是带着解决方案回应
3. 不要主动把话题引向心理健康，除非对方向你发出了求助信号
4. 用户已经聊过的话题不要重复追问——那会让对方觉得你根本没在听

## 需要警惕的情绪信号
以下信号说明对方可能不是在闲聊，而是用轻松的语气在试探性求助：
- 反复提到负面感受（"最近老是这样""其实也没什么""习惯了"）
- 在轻松话题中突然插入沉重内容（"哈哈其实也就是表面开心"）
- 用"就是"来弱化一个明显困扰自己的问题（"就是最近有点睡不着"）
识别到这些信号后，不要立刻切入咨询师模式，而是用一句轻而准的话打开空间：
✓ "你刚才说'表面开心'——那内里呢？" ✗ "听起来你可能有抑郁倾向，我们来聊聊你的情绪问题"

## 对话示例
用户："今天天气真好"
✓ "是啊！你今天有出去转转吗？"
  → 为什么好：自然的闲聊节奏，一个轻松的提问

用户："你知道附近有什么好吃的吗"
✓ "这个我不太确定呢，建议你搜一下本地推荐~"
  → 为什么好：超出能力范围时不硬装，轻松承认

用户："哈哈其实也就是表面开心，心里挺难受的"
✓ "表面开心，心里难受——这种状态一定很消耗吧。想聊聊吗？"
  → 为什么好：挑出对方用的"表面开心"这个表达，镜映回去；用"想聊聊吗"而非"你应该跟我聊聊"
✗ "让我来帮你分析一下你的情绪状态。你最近遇到了什么事情？"
  → 为什么差：太正式、太快进入分析模式，会把对方吓回去
"""

# ── 上下文工程（包 B）：token 预算自适应截断 + 滚动摘要 ──────────
# 旧的固定窗口 MAX_HISTORY_FOR_PROMPT=10 轮已退役：现在按 HISTORY_TOKEN_BUDGET
# 动态选「上下文摘要 + 最近 N 轮原文」，token 计数用 tiktoken（Qwen BPE 的近似，
# 预算留安全余量；tiktoken 不可用则回退字符启发式）。

SUMMARY_SYSTEM_PROMPT = """你是一个心理咨询对话摘要器。把输入的心理咨询对话压缩成一段 2-4 句的中文摘要，保留：
- 用户的核心困扰与关键症状（时间线、频率、严重度、睡眠、精力、身体等线索）
- 已确认的重要事实和对话中的进展或变化
不要编造任何对话中没有的信息。直接输出摘要正文，不要任何前缀、解释或标题。"""

_TIKTOKEN_ENC = None  # 模块级缓存，避免每次 token 计数重复加载编码


class InterventionReplyGenerator:
    """回复生成器：管线结论 → LCEL chain → InterventionResult"""

    def __init__(self, llm=None, retriever=None, session_store=None, summary_llm=None):
        self._llm = llm
        self._retriever = retriever
        self._session_store = session_store
        self._summary_llm = summary_llm  # 滚动摘要专用 LLM；None 时延迟构建轻量模型

    # ── LCEL helper ─────────────────────────────────────────

    def _invoke_chain(self, system_text: str, user_text: str) -> str:
        """构建 LCEL 链：ChatPromptTemplate → LLM → StrOutputParser，返回纯文本回复。

        在调用前应用 prompt 注入防御：
        - system prompt 尾部追加指令层级声明
        - user_text 用 <user_message> 边界标签包裹
        """
        system_text = system_text + INSTRUCTION_HIERARCHY_SUFFIX
        wrapped = wrap_user_text(user_text)

        chain = (
            ChatPromptTemplate.from_messages([
                ("system", system_text),
                ("human", "{user_text}"),
            ])
            | RunnableLambda(lambda msgs: self._llm.invoke(msgs))
            | StrOutputParser()
        )
        return chain.invoke({"user_text": wrapped})

    # ── session helpers (unchanged) ─────────────────────────

    def _get_session_store(self):
        if self._session_store is None:
            from core.memory.session_memory import SessionManager
            self._session_store = SessionManager
        return self._session_store

    def _format_history(self, session_id: Optional[str]) -> str:
        """构建注入 prompt 的对话历史块：上下文摘要 + 最近 N 轮原文。

        包 B 替换旧固定窗口（最近 10 轮）：改为按 HISTORY_TOKEN_BUDGET 动态选轮。
        块结构（预算优先覆盖摘要，余量给最近原文，至少保留一轮完整对话）：
            【上下文摘要】
            {get_context_summary() 事实摘要}       ← 接线既有 get_context_summary()
            {早期对话摘要：LLM 滚动摘要}            ← rolling_summary（对标 MemGPT）
            【最近对话】
            用户：... / 助手：...                  ← 预算内自最近往早选完整轮次
        """
        if not session_id:
            return "（无历史对话，这是第一轮）"
        try:
            session = self._get_session_store().get_session(session_id)
            history = session.get_history_for_prompt()
            logger.info(f"[HISTORY_LOAD] session={session_id} turns={len(history)//2 if history else 0}")
        except Exception as e:
            logger.warning(f"[HISTORY_LOAD] FAILED session={session_id}: {e}")
            return "（无法获取对话历史）"
        if not history:
            logger.warning(f"[HISTORY_LOAD] session={session_id} EMPTY history")
            return "（无历史对话，这是第一轮）"

        # 摘要前缀：事实摘要（get_context_summary）+ LLM 滚动摘要。任一步失败安全降级为空。
        prefix_parts = []
        try:
            context_summary = session.get_context_summary()
            if context_summary:
                prefix_parts.append(context_summary)
        except Exception:
            pass
        try:
            rolling = getattr(session.metadata, "rolling_summary", "") or ""
            if rolling:
                prefix_parts.append(f"早期对话摘要：{rolling}")
        except Exception:
            pass
        prefix_text = "\n".join(prefix_parts)

        # token 预算自适应选轮：预算先覆盖摘要前缀，余量给最近原文（至少留 256 token）
        budget = settings.HISTORY_TOKEN_BUDGET
        prefix_tokens = self._count_tokens(prefix_text)
        recent_budget = max(budget - prefix_tokens, 256)
        recent, used = self._select_recent_within_budget(history, recent_budget)
        logger.info(
            "[HISTORY_FORMAT] session=%s turns=%d used_tokens=%d budget=%d prefix_tokens=%d",
            session_id, len(recent) // 2, used, budget, prefix_tokens,
        )

        lines = []
        if prefix_text:
            lines.append("【上下文摘要】")
            lines.append(prefix_text)
        lines.append("【最近对话】")
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    # ── 上下文工程 helpers：token 计数 + 预算选轮 ─────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        """近似 token 计数：优先 tiktoken cl100k_base（Qwen BPE 的近似）。

        tiktoken 不可用（离线/导入失败）时回退字符启发式：CJK 字符按 1 token/字，
        其余按 4 字符 1 token（向上取整）。启发式偏保守（偏多），对预算上限安全。
        """
        global _TIKTOKEN_ENC
        if not text:
            return 0
        try:
            if _TIKTOKEN_ENC is None:
                import tiktoken
                _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
            return len(_TIKTOKEN_ENC.encode(text))
        except Exception:
            cjk = sum(
                1 for ch in text
                if "一" <= ch <= "鿿" or "　" <= ch <= "〿"
            )
            other = len(text) - cjk
            return cjk + (other + 3) // 4

    def _select_recent_within_budget(
        self, history: List[Dict[str, str]], budget_tokens: int
    ) -> tuple[List[Dict[str, str]], int]:
        """按 token 预算自最近往早选完整对话轮次。

        至少保留最近一轮（预算再紧也返回非空）；随后逐条向前加入直到超预算。
        若最旧一条是助手回复（截断了该轮的用户原话），丢弃它保持「用户→助手」配对完整。
        返回 (kept_messages, used_tokens)。
        """
        kept: List[Dict[str, str]] = []
        used = 0
        for msg in reversed(history):
            cost = self._count_tokens(msg["content"])
            if kept and used + cost > budget_tokens:
                break
            kept.append(msg)
            used += cost
        kept.reverse()
        if kept and kept[0]["role"] == "assistant":
            kept.pop(0)
        return kept, used

    # ── 滚动摘要（对标 MemGPT working/summary）────────────────

    @staticmethod
    def _build_summary_input(previous_summary: str, recent_history: List[Dict[str, str]]) -> str:
        """构建滚动摘要的 LLM 输入：已有摘要 + 最近几轮原文，要求合并。"""
        parts = []
        if previous_summary:
            parts.append(f"已有摘要：\n{previous_summary}\n")
        parts.append("需要压缩的新对话：")
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            parts.append(f"{role}：{msg['content']}")
        parts.append("请输出合并后的新摘要（保留旧摘要关键信息，补充新对话内容）：")
        return "\n".join(parts)

    def _get_summary_llm(self):
        """延迟构建滚动摘要专用 LLM（轻量模型 + enable_thinking=false）。"""
        if self._summary_llm is None:
            from core.llm.base import get_llm_adapter, LLMConfig

            cfg = LLMConfig(
                model_name=settings.SUMMARY_MODEL_NAME,
                temperature=0,
                max_tokens=400,
                model_kwargs={"extra_body": {"enable_thinking": False}},
            )
            self._summary_llm = get_llm_adapter("qwen", config=cfg).llm
        return self._summary_llm

    def _invoke_summary(self, prompt: str) -> str:
        """调用摘要 LLM 生成滚动摘要，失败/空回复返回空串（安全降级）。"""
        try:
            llm = self._get_summary_llm()
            reply = llm.invoke(
                [SystemMessage(content=SUMMARY_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            if isinstance(reply, str):
                return reply.strip()
            return str(getattr(reply, "content", "")).strip()
        except Exception as e:
            logger.warning("[ROLLING_SUMMARY] LLM 调用失败: %s", e)
            return ""

    def _maybe_roll_summary(self, session_id: Optional[str]) -> None:
        """每 SUMMARY_EVERY_N_TURNS 轮把旧对话压成滚动摘要。

        仅在普通对话路径触发：本方法只在 _save_turn 内调用，而危机路径走
        service._save_crisis_turn（轻量 SessionManager，绝不触碰 generator）——
        危机零延迟铁律由调用链天然保证。
        失败不阻断：摘要 LLM 失败 / 会话异常只记 warning，绝不影响本轮回复保存。
        """
        if not session_id:
            return
        interval = settings.SUMMARY_EVERY_N_TURNS
        if interval <= 0:
            return
        try:
            session = self._get_session_store().get_session(session_id)
            turns = session.metadata.message_count // 2
            if turns < interval or turns % interval != 0:
                return
            previous = session.metadata.rolling_summary or ""
            history = session.get_history_for_prompt()
            recent = history[-(interval * 2):]
            prompt = self._build_summary_input(previous, recent)
            summary = self._invoke_summary(prompt)
            if summary:
                session.set_rolling_summary(summary, last_turn=turns)
                logger.info(
                    "[ROLLING_SUMMARY] session=%s turns=%d summary_len=%d",
                    session_id, turns, len(summary),
                )
        except Exception as e:
            logger.warning("[ROLLING_SUMMARY] FAILED session=%s: %s", session_id, e)

    def _get_probed_dimensions_text(self, session_id: Optional[str]) -> str:
        """构建"已探测/未探测维度"提示文本，用于注入 system prompt。

        告知 LLM 哪些临床维度已经问过（禁止再问），哪些还没聊过（可以自然探）。
        全部探测完毕时提示停止探测、只做共情陪伴。
        """
        if not session_id:
            return ""
        try:
            session = self._get_session_store().get_session(session_id)
            probed = session.get_probed_dimensions()
        except Exception:
            return ""
        if not probed:
            return "**探测进度**：尚未探测任何维度，你可以从任意维度开始。"

        remaining = [d for d in PROBE_DIMENSIONS if d not in probed]
        probed_text = "、".join(probed)
        if remaining:
            remaining_text = "、".join(remaining)
            return (
                f"**探测进度**：已探测 — {probed_text}。"
                f"未探测 — {remaining_text}。"
                "只能从「未探测」中选一个方向轻轻探一下，已探测的绝不要重复问。"
            )
        else:
            return (
                f"**探测进度**：全部 6 个维度已探测完毕（{probed_text}）。"
                "不要再探测任何维度，只需共情陪伴，让对方感到被理解。"
            )

    # ── 医生模式 helper ──────────────────────────────────────

    def _get_doctor_prompt(self, route: str) -> Optional[str]:
        """DOCTOR_MODE=true 时返回对应路由的周医生 prompt 模板。"""
        if not settings.DOCTOR_MODE:
            return None
        try:
            from modules.intervention.persona import get_doctor_prompt
            return get_doctor_prompt(route, settings.DOCTOR_PERSONA)
        except Exception as e:
            logger.warning("加载 doctor prompt 失败: %s，回退原 prompt", e)
            return None

    def _get_safety_probe(self, req: InterventionRequest) -> str:
        """verdict==probe 时构建安全探针指令段，否则空串。

        探针由语义安全评估器生成（LLM probe_suggestion），规则提供兜底模板。
        医生以唯一声音自然问出，不打断 persona（ADR-0013 2-agent 边界）。
        """
        verdict_dict = req.safety_verdict or {}
        if verdict_dict.get("verdict") != "probe":
            return ""
        suggestion = verdict_dict.get("probe_suggestion")
        if not suggestion:
            from modules.assessment.safety_judge import FALLBACK_PROBE
            suggestion = FALLBACK_PROBE.get(
                verdict_dict.get("risk_type", "general"),
                FALLBACK_PROBE["general"],
            )
        from modules.intervention.persona import ZHOU_SAFETY_PROBE_TEMPLATE
        return ZHOU_SAFETY_PROBE_TEMPLATE.format(
            risk_type=verdict_dict.get("risk_type", "general"),
            probe_suggestion=suggestion,
        )

    @staticmethod
    def _format_scid_directive(scid_directive: str) -> str:
        """把 SCID 访谈引擎的指令格式化为追加到 system prompt 的指令块。

        为空时返回空串（不改变原 prompt）。追加在 INSTRUCTION_HIERARCHY_SUFFIX 之前，
        与安全探针同模式：格式化的模板 prompt 之后追加，避免改动各 persona 模板。
        """
        if not scid_directive:
            return ""
        return (
            "\n\n## SCID-5 结构化访谈指令（评估引擎引导，本轮有效）\n"
            + scid_directive
            + "\n要求：用周医生一贯温暖、口语化的语气完成上述指令，"
              "绝不暴露『我在按手册访谈』；一次只问一个问题；"
              "不要重复用户已经回答过的内容；"
              "若用户情绪强烈或涉及安全风险，优先共情与安全，可暂停访谈。"
        )

    def _build_assessor_context(self, session_id: Optional[str]) -> str:
        """从 session 构建评估上下文文本，注入 prompt。"""
        if not session_id:
            return "（无评估上下文，这是第一轮对话）"
        try:
            session = self._get_session_store().get_session(session_id)
            return session.get_assessor_context()
        except Exception:
            return "（无法获取评估上下文）"

    def _get_format_kwargs(
        self, req: InterventionRequest, knowledge_text: str = ""
    ) -> Dict[str, Any]:
        """构建 prompt 格式化参数。DOCTOR_MODE 和普通模式统一入口。"""
        emotion = req.emotion or {}
        kw: Dict[str, Any] = {
            "primary_emotion": emotion.get("primary_emotion", "neutral"),
            "intensity": emotion.get("intensity", 0.5),
            "risk": emotion.get("risk", 0.0),
            "conversation_history": self._format_history(req.session_id),
            "probed_dimensions_note": self._get_probed_dimensions_text(req.session_id),
        }
        if settings.DOCTOR_MODE:
            kw["assessor_context"] = self._build_assessor_context(req.session_id)
            kw["phase"] = self._get_session_phase(req.session_id)
            kw["zhou_style_refs"] = self._get_zhou_style_refs(req.user_text)
        if knowledge_text:
            kw["retrieved_knowledge"] = knowledge_text
        return kw

    def _get_zhou_style_refs(self, user_text: str) -> str:
        """DOCTOR_MODE 下检索周医生风格参考，格式化为 prompt 注入段。

        索引未建 / 检索失败 → 返回空串，安全降级（模板占位符替换为空）。
        """
        if not user_text:
            return ""
        try:
            from modules.intervention.rag.zhou_style import get_zhou_style_retriever
            retriever = get_zhou_style_retriever()
            hits = retriever.retrieve(user_text, top_k=3)
            return retriever.format_for_prompt(hits, max_items=2)
        except Exception:
            logger.warning("ZhouStyle 风格参考获取失败，跳过", exc_info=True)
            return ""

    def _get_session_phase(self, session_id: Optional[str]) -> str:
        """读取 session 的当前阶段。"""
        if not session_id:
            return "check_in"
        try:
            session = self._get_session_store().get_session(session_id)
            return session.metadata.phase
        except Exception:
            return "check_in"

    def _save_probed_dimension(self, session_id: Optional[str], reply_text: str) -> None:
        """检测本轮回复探测了哪个维度并持久化。"""
        if not session_id:
            return
        dim = detect_probed_dimension(reply_text)
        if dim is None:
            return
        try:
            session = self._get_session_store().get_session(session_id)
            session.add_probed_dimension(dim)
        except Exception:
            pass

    def _save_turn(self, session_id: Optional[str], user_text: str, ai_reply: str) -> None:
        if not session_id:
            return
        try:
            session = self._get_session_store().get_session(session_id)
            session.add_user_message(user_text)
            session.add_ai_message(ai_reply)
            logger.info(f"[TURN_SAVE] session={session_id} user_msg_len={len(user_text)} ai_msg_len={len(ai_reply)}")
            # 滚动摘要：仅在普通对话路径触发（crisis 走 service._save_crisis_turn 不经过这里）。
            # 摘要 LLM 调用失败只记 warning，绝不影响本轮回复保存。
            self._maybe_roll_summary(session_id)
        except Exception as e:
            logger.warning(f"[TURN_SAVE] FAILED session={session_id}: {e}")

    # ── streaming (async) methods ──────────────────────────

    async def astream_comfort(self, req: InterventionRequest,
                              scid_directive: str = "") -> AsyncIterator[str]:
        """安抚路由流式生成：绕过 LCEL，直接调用 self._llm.astream()"""
        doctor_prompt = self._get_doctor_prompt("comfort")
        format_kw = self._get_format_kwargs(req)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
        else:
            system_text = COMFORT_SYSTEM_PROMPT.format(**format_kw)
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
            + INSTRUCTION_HIERARCHY_SUFFIX
        )
        wrapped = wrap_user_text(req.user_text)

        messages = [SystemMessage(content=system_text), HumanMessage(content=wrapped)]
        full_reply: list[str] = []

        async for chunk in self._llm.astream(messages):
            full_reply.append(chunk)
            yield chunk

        reply_text = "".join(full_reply)
        self._save_turn(req.session_id, req.user_text, reply_text)
        self._save_probed_dimension(req.session_id, reply_text)

    async def astream_general(self, req: InterventionRequest,
                              scid_directive: str = "") -> AsyncIterator[str]:
        """通用路由流式生成"""
        doctor_prompt = self._get_doctor_prompt("general")
        format_kw = self._get_format_kwargs(req)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
        else:
            system_text = GENERAL_SYSTEM_PROMPT.format(**format_kw)
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
            + INSTRUCTION_HIERARCHY_SUFFIX
        )
        wrapped = wrap_user_text(req.user_text)

        messages = [SystemMessage(content=system_text), HumanMessage(content=wrapped)]
        full_reply: list[str] = []

        async for chunk in self._llm.astream(messages):
            full_reply.append(chunk)
            yield chunk

        reply_text = "".join(full_reply)
        self._save_turn(req.session_id, req.user_text, reply_text)
        self._save_probed_dimension(req.session_id, reply_text)

    async def astream_knowledge(self, req: InterventionRequest,
                                 enriched_query: str = None,
                                 scid_directive: str = "") -> AsyncIterator[str]:
        """知识路由流式生成：RAG 检索在流式开始前同步完成"""
        emotion = req.emotion or {}

        if enriched_query:
            query = enriched_query
        else:
            query = req.user_text
            if emotion.get("primary_emotion"):
                query = f"{emotion['primary_emotion']} {query}"

        docs = []
        if self._retriever:
            docs = self._retriever.retrieve(query, top_k=3)
        knowledge_text = "\n".join(f"- {d}" for d in docs) if docs else "（知识库暂无相关内容）"

        doctor_prompt = self._get_doctor_prompt("knowledge")
        format_kw = self._get_format_kwargs(req, knowledge_text=knowledge_text)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
        else:
            system_text = KNOWLEDGE_SYSTEM_PROMPT.format(**format_kw)
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
            + INSTRUCTION_HIERARCHY_SUFFIX
        )
        wrapped = wrap_user_text(req.user_text)

        messages = [SystemMessage(content=system_text), HumanMessage(content=wrapped)]
        full_reply: list[str] = []

        async for chunk in self._llm.astream(messages):
            full_reply.append(chunk)
            yield chunk

        reply_text = "".join(full_reply)
        self._save_turn(req.session_id, req.user_text, reply_text)
        self._save_probed_dimension(req.session_id, reply_text)

    # ── generate methods ────────────────────────────────────

    def generate_comfort(self, req: InterventionRequest,
                         scid_directive: str = "") -> InterventionResult:
        doctor_prompt = self._get_doctor_prompt("comfort")
        format_kw = self._get_format_kwargs(req)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
            meta = {"implementation": "llm_comfort_doctor"}
        else:
            system_text = COMFORT_SYSTEM_PROMPT.format(**format_kw)
            meta = {"implementation": "llm_comfort"}
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
        )
        reply = self._invoke_chain(system_text, req.user_text)
        self._save_turn(req.session_id, req.user_text, reply)
        self._save_probed_dimension(req.session_id, reply)
        return InterventionResult(
            reply=reply, empathy="", suggestion="", action_items=[],
            chain_of_thought=None, emergency_triggered=False,
            meta=meta,
        )

    def generate_general(self, req: InterventionRequest,
                         scid_directive: str = "") -> InterventionResult:
        doctor_prompt = self._get_doctor_prompt("general")
        format_kw = self._get_format_kwargs(req)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
            meta = {"implementation": "llm_general_doctor"}
        else:
            system_text = GENERAL_SYSTEM_PROMPT.format(**format_kw)
            meta = {"implementation": "llm_general"}
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
        )
        reply = self._invoke_chain(system_text, req.user_text)
        self._save_turn(req.session_id, req.user_text, reply)
        self._save_probed_dimension(req.session_id, reply)
        return InterventionResult(
            reply=reply, empathy="", suggestion="", action_items=[],
            chain_of_thought=None, emergency_triggered=False,
            meta=meta,
        )

    def generate_knowledge(self, req: InterventionRequest, enriched_query: str = None,
                           scid_directive: str = "") -> InterventionResult:
        emotion = req.emotion or {}

        if enriched_query:
            query = enriched_query
        else:
            query = req.user_text
            if emotion.get("primary_emotion"):
                query = f"{emotion['primary_emotion']} {query}"

        docs = []
        if self._retriever:
            docs = self._retriever.retrieve(query, top_k=3)
        knowledge_text = "\n".join(f"- {d}" for d in docs) if docs else "（知识库暂无相关内容）"

        doctor_prompt = self._get_doctor_prompt("knowledge")
        format_kw = self._get_format_kwargs(req, knowledge_text=knowledge_text)
        if doctor_prompt:
            system_text = doctor_prompt.format(**format_kw)
            meta = {"implementation": "llm_knowledge_doctor", "retrieved_docs": len(docs)}
        else:
            system_text = KNOWLEDGE_SYSTEM_PROMPT.format(**format_kw)
            meta = {"implementation": "llm_knowledge", "retrieved_docs": len(docs)}
        system_text = (
            system_text + self._get_safety_probe(req)
            + self._format_scid_directive(scid_directive)
        )
        reply = self._invoke_chain(system_text, req.user_text)
        self._save_turn(req.session_id, req.user_text, reply)
        self._save_probed_dimension(req.session_id, reply)
        return InterventionResult(
            reply=reply, empathy="",
            suggestion=f"知识来源：基于检索到的 {len(docs)} 条心理学参考资料",
            action_items=[], chain_of_thought=None, emergency_triggered=False,
            meta=meta,
        )
