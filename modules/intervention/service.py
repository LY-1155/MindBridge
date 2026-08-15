"""InterventionService：干预闭环模块入口，按路由分支调度"""

from __future__ import annotations
import logging
from typing import Optional, AsyncIterator

from config.settings import settings
from schemas.contracts import InterventionRequest, InterventionResult
from modules.intervention.crisis_handler import CrisisHandler

logger = logging.getLogger(__name__)


class InterventionService:
    """干预模块主入口。按 route 分支：crisis → CrisisHandler，comfort → Generator，
    knowledge → ScaleOrchestrator（量表进行中/可触发）→ Generator + RAG（默认）"""

    def __init__(
        self,
        crisis_handler: Optional[CrisisHandler] = None,
        generator=None,  # InterventionReplyGenerator，延迟 import 避免循环
        retriever=None,  # KnowledgeRetriever，延迟 import 避免循环
        llm_adapter=None,  # BaseLLMAdapter，延迟 import 避免循环
        orchestrator=None,  # ScaleOrchestrator
    ):
        self._crisis = crisis_handler
        self._generator = generator
        self._retriever = retriever
        self._llm_adapter = llm_adapter
        self._orchestrator = orchestrator

    def _get_crisis(self) -> CrisisHandler:
        if self._crisis is None:
            self._crisis = CrisisHandler()
        return self._crisis

    def _get_generator(self):
        if self._generator is None:
            from modules.intervention.generator import InterventionReplyGenerator
            from modules.intervention.rag.retriever import get_knowledge_retriever
            from core.llm.base import get_llm_adapter, LLMConfig
            from core.rag.query_rewriter import QueryRewriter
            from config.settings import settings as app_settings

            llm = self._llm_adapter or get_llm_adapter("qwen")
            retriever = self._retriever or get_knowledge_retriever()
            # 启用查询改写器（使用独立模型配置）
            if getattr(retriever, '_rewriter', None) is None:  # noqa: SLF001
                rewriter_config = LLMConfig(
                    model_name=app_settings.REWRITER_MODEL_NAME,
                    temperature=0,
                    max_tokens=1024,  # 推理模型需要足够空间（reasoning ~512 + output）
                )
                rewriter_llm = get_llm_adapter("openai_compatible", config=rewriter_config)
                retriever._rewriter = QueryRewriter(rewriter_llm.llm)  # noqa: SLF001

            # 启用本地 BGE cross-encoder 重排序（零 API 费用；qwen3-rerank API 免费额度已耗尽）
            if getattr(retriever, '_reranker', None) is None:  # noqa: SLF001
                from core.rag.reranker import BGEReranker
                retriever._reranker = BGEReranker(top_n=20, top_k=3)  # noqa: SLF001

            self._generator = InterventionReplyGenerator(llm=llm, retriever=retriever)
        return self._generator

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from modules.intervention.scale.orchestrator import ScaleOrchestrator
            from modules.intervention.scale.scorer import ScaleScorer
            from core.llm.base import get_llm_adapter

            llm = self._llm_adapter or get_llm_adapter("qwen")
            # scorer 不传 llm，让它自己建轻量模型（默认 qwen-turbo）
            self._orchestrator = ScaleOrchestrator(llm=llm, scorer=ScaleScorer())
        return self._orchestrator

    def _get_session(self, session_id: Optional[str]):
        if not session_id:
            return None
        from core.memory.session_memory import SessionManager
        return SessionManager.get_session(session_id)

    def _save_crisis_turn(self, session_id: Optional[str], user_text: str, ai_reply: str) -> None:
        """危机路径消息落库：用户高风险原话 + AI 危机话术（安全审计）。

        修复前危机路由只发模板不落库——用户的高风险表述和 AI 危机回复
        （热线话术、rescue 触发）在 messages 表里没有留痕，仅靠
        safety_flags.matched_terms 间接记录。此方法不用 _get_generator()
        （会初始化 LLM/检索器，危机路径要求零延迟），直接走轻量 SessionManager。
        """
        if not session_id:
            return
        try:
            session = self._get_session(session_id)
            if session is None:
                return
            session.add_user_message(user_text)
            session.add_ai_message(ai_reply)
            logger.info(
                "[TURN_SAVE] crisis session=%s user_msg_len=%d ai_msg_len=%d",
                session_id, len(user_text), len(ai_reply),
            )
        except Exception as e:
            logger.warning("[TURN_SAVE] crisis FAILED session=%s: %s", session_id, e)

    def _get_assessor(self):
        """延迟初始化 FamilySystemAssessor。"""
        if not hasattr(self, "_assessor"):
            from modules.assessment.family_assessor import FamilySystemAssessor
            self._assessor = FamilySystemAssessor()
        return self._assessor

    def _get_scid_tracker(self):
        """延迟初始化 SCIDTracker。"""
        if not hasattr(self, "_scid_tracker"):
            from modules.assessment.scid_tracker import SCIDTracker
            self._scid_tracker = SCIDTracker()
        return self._scid_tracker

    # ── 医生模式：共享评估逻辑 ──────────────────────────────

    def _run_doctor_assessment(
        self, req: InterventionRequest, session, route: str
    ) -> tuple[str, Optional[str], str]:
        """DOCTOR_MODE 共享评估：返回 (possibly_upgraded_route, enriched_query, scid_directive)。

        危机判定整合（ADR-0013）：风险词命中 ≠ 危机触发。
        P0 硬闸门 + 语义安全评估器三级裁决驱动最终路由。

        scid_directive：主动式 SCID 访谈引擎产出的「下一步该问什么」指令
        （仅抑郁模块起步，见 modules/assessment/scid_interview.py）。引擎纯规则、
        不调 LLM；指令由 generator 注入 prompt，用医生语气自然问出。
        """
        if not settings.DOCTOR_MODE or session is None:
            return route, None, ""

        scid_enriched_query: Optional[str] = None
        scid_directive: str = ""
        try:
            assessor = self._get_assessor()
            assess_result = assessor.assess(
                user_text=req.user_text,
                message_count=session.metadata.message_count,
                existing_phase=session.metadata.phase,
                existing_hypothesis=session.metadata.working_hypothesis,
                existing_members=session.metadata.family_members,
                emotion=req.emotion or {},
                route=req.route or {},
            )

            if assess_result.suggested_phase != session.metadata.phase:
                session.update_phase(assess_result.suggested_phase)
            if assess_result.hypothesis_update is not None:
                session.update_hypothesis(assess_result.hypothesis_update)
            for member in assess_result.new_family_members:
                session.add_family_member(
                    role=member["role"], label=member.get("label", "")
                )

            tracker = self._get_scid_tracker()
            scid_update = tracker.update(
                user_text=req.user_text,
                existing_flags=session.metadata.scid_flags,
            )
            session.update_scid_flags(scid_update.criteria_met)

            # ── 主动式 SCID 访谈（医生模式，抑郁模块起步）──
            # 纯规则状态机：决定下一句问什么，产出 directive 由 generator 注入 prompt。
            # 失败不阻断（回退到无访谈指令的正常医生对话）。
            try:
                from modules.assessment.scid_interview import SCIDInterviewEngine

                engine = SCIDInterviewEngine()
                passive_mdd = (
                    session.metadata.scid_flags or {}
                ).get("MDD", {}).get("criteria_met", [])
                iv_state, scid_directive = engine.step_turn(
                    session.metadata.scid_interview_state,
                    req.user_text,
                    passive_mdd,
                )
                if iv_state is not None:
                    session.update_scid_interview_state(iv_state)
                if scid_directive:
                    logger.info(
                        "[DOCTOR_MODE] SCID interview active step=%s confirmed=%s denied=%s",
                        iv_state.get("step") if iv_state else "-",
                        (iv_state or {}).get("criteria_confirmed"),
                        (iv_state or {}).get("criteria_denied"),
                    )
            except Exception:
                logger.warning(
                    "[DOCTOR_MODE] SCID interview engine failed", exc_info=True
                )

            # 注：scid risk_flags 不再直接升级路由 —— 改为语义安全评估器的锚点输入。
            # 评估器已在图上读取 session.scid_flags 做跨轮累积判断（复现 → 持续监测）。
            if scid_update.suggested_retrieval_query:
                scid_enriched_query = scid_update.suggested_retrieval_query

            # ── 危机判定整合 ──
            route = self._apply_safety_verdict(req, session, route, assess_result)

            logger.debug(
                "[DOCTOR_MODE] phase=%s probe=%s verdict=%s scid_directive=%s",
                assess_result.suggested_phase,
                assess_result.probe_direction,
                (req.safety_verdict or {}).get("verdict"),
                bool(scid_directive),
            )
        except Exception:
            logger.warning("[DOCTOR_MODE] Assessment failed", exc_info=True)

        return route, scid_enriched_query, scid_directive

    def _apply_safety_verdict(
        self, req: InterventionRequest, session, route: str, assess_result
    ) -> str:
        """危机判定整合：P0 硬闸门 + 评估器三级裁决 → 最终路由。

        - P0 硬闸门（family_assessor 升级 / safety.blocked）→ crisis（不可降级）
        - verdict crisis → crisis
        - verdict probe → 状态机迁移（确认→CRISIS / 否认→NONE / 累积→CRISIS），
          未升级则保持路由（generator 注入安全探针）
        - verdict no_risk → 正常对话（router 的 crisis 降级到 comfort 共情路径）
        - verdict None（评估器跳过/失败）→ 保守保留 router 决策
        """
        verdict_dict = req.safety_verdict or {}
        verdict = verdict_dict.get("verdict")
        safety = req.safety or {}

        # 1. P0 硬闸门
        if assess_result.escalation_flag or safety.get("blocked"):
            session.metadata.safety_state = {
                "status": "CRISIS", "probe_count": 0, "denial_mark": False,
            }
            self._persist_safety_state(session)
            logger.info("[DOCTOR_MODE] P0 硬闸门 → crisis")
            return "crisis"

        # 2. 状态机迁移（crisis / probe / no_risk 三种裁决）
        if verdict in ("crisis", "probe", "no_risk"):
            from modules.assessment.risk_anchors import transition_safety_state

            # 只有"有风险锚点"的 probe 才计入累积升级；纯情绪宣泄（无锚点）
            # 反复出现不触发危机（收窄原则，ADR-0013）
            anchored_probe = bool(verdict_dict.get("matched_anchors"))
            new_state = transition_safety_state(
                session.metadata.safety_state,
                verdict,
                req.user_text,
                max_probe_count=settings.SAFETY_PROBE_MAX_COUNT,
                anchored_probe=anchored_probe,
            )
            session.metadata.safety_state = new_state
            self._persist_safety_state(session)

            if new_state["status"] == "CRISIS":
                logger.info(
                    "[DOCTOR_MODE] 状态机 → CRISIS (verdict=%s probe_count=%d)",
                    verdict, new_state["probe_count"],
                )
                return "crisis"

        # 3. 非 crisis 裁决下的路由处理
        if verdict == "no_risk":
            if route == "crisis":
                logger.info("[DOCTOR_MODE] 评估器 no_risk 降级 router crisis → comfort")
                return "comfort"
            return route
        if verdict == "probe":
            # probe：医生继续对话并注入安全探针（generator 读 req.safety_verdict）
            if route == "crisis":
                return "comfort"
            return route

        # 4. verdict None（评估器跳过/失败）→ 保守保留 router 决策
        return route

    @staticmethod
    def _persist_safety_state(session) -> None:
        """持久化 safety_state。持久化失败不阻断（内存兜底，下次仍可用）。"""
        try:
            session.update_safety_state(session.metadata.safety_state)
        except Exception:
            pass

    def _resolve_route(self, req: InterventionRequest) -> str:
        """处理低置信度回退：knowledge/general + confidence<0.5 → 退回上一级"""
        route_dict = req.route or {}
        route = route_dict.get("route") or "general"
        confidence = route_dict.get("confidence", 1.0)

        # crisis 路由不降级，安全优先
        if route == "crisis":
            return route

        # knowledge 低置信度 → 退回 comfort
        if route == "knowledge" and confidence < 0.5:
            return "comfort"

        # general 低置信度 → 退回 comfort
        if route == "general" and confidence < 0.5:
            return "comfort"

        return route

    def intervene(self, req: InterventionRequest) -> InterventionResult:
        route = self._resolve_route(req)

        # 量表进行中：不管路由走到哪，优先继续量表流程
        session = self._get_session(req.session_id)
        if session is not None:
            scale_state = session.metadata.scale_state
            if scale_state and scale_state.get("status") == "in_progress":
                orch = self._get_orchestrator()
                result = orch.process_turn(req.user_text, session)
                if result.is_complete:
                    completed = session.metadata.scale_state
                    if completed and completed.get("status") == "completed":
                        session.metadata.scale_history.append({
                            "scale_name": completed.get("scale_name"),
                            "total_score": completed.get("total_score"),
                            "level": completed.get("level"),
                            "escalation_flag": completed.get("escalation_flag", False),
                        })

                    # 写入 scale_screenings 表
                    if completed:
                        try:
                            from core.memory.db_storage import DatabaseStorage
                            DatabaseStorage.save_scale_screening(
                                session_id=req.session_id,
                                scale_type=completed.get("scale_name", "unknown"),
                                state=completed.get("status", "completed"),
                                scores=completed.get("scores"),
                                total_score=float(completed.get("total_score", 0)) if completed.get("total_score") is not None else None,
                                user_id=req.user_id or "",
                            )
                        except Exception:
                            pass  # 非关键路径

                    # D10: 串行执行 — 检查 pending 队列，自动启动下一个量表
                    remaining = (completed or {}).get("pending_scales", [])
                    rejection_kw = ["不要", "先不谈", "太多了", "先不"]
                    should_abandon = any(kw in req.user_text for kw in rejection_kw)
                    if remaining and not should_abandon:
                        next_scale = remaining[0]
                        next_pending = remaining[1:]
                        next_reply = orch.start(next_scale, session)
                        if next_pending:
                            ns = session.metadata.scale_state
                            if ns:
                                ns["pending_scales"] = next_pending
                                session.metadata.scale_state = ns
                        return InterventionResult(
                            reply=result.reply + "\n\n" + next_reply,
                            empathy="",
                            suggestion="",
                            action_items=[],
                            chain_of_thought=None,
                            emergency_triggered=result.escalation_flag,
                            meta={
                                "implementation": "scale_screening",
                                "is_complete": False,
                                "scale": next_scale,
                                "level": result.level,
                                "escalation_flag": result.escalation_flag,
                            },
                        )
                    elif remaining and should_abandon:
                        completed["pending_scales"] = []
                        session.metadata.scale_state = completed
                        for sn in remaining:
                            session.metadata.scale_history.append({
                                "scale_name": sn,
                                "status": "abandoned",
                            })
                            try:
                                from core.memory.db_storage import DatabaseStorage
                                DatabaseStorage.save_scale_screening(
                                    session_id=req.session_id,
                                    scale_type=sn,
                                    state="abandoned",
                                    user_id=req.user_id or "",
                                )
                            except Exception:
                                pass

                return InterventionResult(
                    reply=result.reply,
                    empathy="",
                    suggestion="",
                    action_items=[],
                    chain_of_thought=None,
                    emergency_triggered=result.escalation_flag,
                    meta={
                        "implementation": "scale_screening",
                        "is_complete": result.is_complete,
                        "level": result.level,
                        "escalation_flag": result.escalation_flag,
                    },
                )

        # ── 医生模式 ──
        route, scid_enriched_query, scid_directive = self._run_doctor_assessment(
            req, session, route
        )

        # INTERVENTION TRACE — 分发分支
        logger.info("[PIPELINE:TRACE] INTERVENTION dispatch → %s", route)

        if route == "crisis":
            result = self._get_crisis().handle(req)
            self._save_crisis_turn(req.session_id, req.user_text, result.reply)
            return result

        if route == "general":
            return self._get_generator().generate_general(
                req, scid_directive=scid_directive
            )

        if route == "comfort":
            return self._get_generator().generate_comfort(
                req, scid_directive=scid_directive
            )

        if route == "knowledge":
            session = self._get_session(req.session_id)
            orch = self._get_orchestrator()
            # DOCTOR_MODE：跳过量表自动触发，走周医生自然对话（RAG + persona）。
            # 量表仍是后台能力，但家庭对话不主动打断节奏。
            if session is not None and orch is not None and not settings.DOCTOR_MODE:
                # 可触发量表：LLM 语义匹配 → 列表，D10 串行执行
                triggered_scales = orch.should_trigger(req.user_text, req.emotion or {})
                if triggered_scales:
                    scale_name = triggered_scales[0]
                    reply = orch.start(scale_name, session)
                    # 写入 scale_screenings（in_progress）
                    try:
                        from core.memory.db_storage import DatabaseStorage
                        DatabaseStorage.save_scale_screening(
                            session_id=req.session_id,
                            scale_type=scale_name,
                            state="in_progress",
                            user_id=req.user_id or "",
                        )
                    except Exception:
                        pass
                    pending = triggered_scales[1:]
                    if pending:
                        state_raw = session.metadata.scale_state
                        if state_raw:
                            state_raw["pending_scales"] = pending
                            session.metadata.scale_state = state_raw
                    # 保存本轮对话（量表路径也需持久化历史）
                    self._get_generator()._save_turn(req.session_id, req.user_text, reply)
                    self._get_generator()._save_probed_dimension(req.session_id, reply)
                    return InterventionResult(
                        reply=reply,
                        empathy="",
                        suggestion="",
                        action_items=[],
                        chain_of_thought=None,
                        emergency_triggered=False,
                        meta={"implementation": "scale_screening", "status": "started", "scale": scale_name},
                    )

            # 默认：RAG + LLM（量表结果反哺知识检索）
            # 医生模式：优先使用 SCID tracker 生成的精准 query
            if scid_enriched_query:
                enriched_query = scid_enriched_query
            else:
                enriched_query = req.user_text
                if session is not None:
                    from modules.intervention.rag.scale_feedback import enrich_query_with_scale
                    enriched_query = enrich_query_with_scale(req.user_text, session)
            return self._get_generator().generate_knowledge(
                req, enriched_query=enriched_query, scid_directive=scid_directive
            )

        # 未知路由，返回占位
        return InterventionResult(
            reply="[placeholder] 待实现",
            empathy="",
            suggestion="",
            action_items=[],
            chain_of_thought=None,
            emergency_triggered=False,
            meta={"implementation": "stub"},
        )

    async def astream_intervene(self, req: InterventionRequest) -> AsyncIterator[str]:
        """流式干预入口。复制 intervene() 的路由分派逻辑，但以 AsyncIterator 产出 token。
        量表进行中 / crisis 路由无需 LLM 调用，yield 完整回复为单个 chunk。
        """
        route = self._resolve_route(req)

        # 量表进行中：不管路由走到哪，优先继续量表流程（非 LLM 调用，yield 单 chunk）
        session = self._get_session(req.session_id)
        if session is not None:
            scale_state = session.metadata.scale_state
            logger.info("[SCALE:CHECK] session=%s scale_state=%s",
                         req.session_id,
                         scale_state)
            if scale_state and scale_state.get("status") == "in_progress":
                orch = self._get_orchestrator()
                result = orch.process_turn(req.user_text, session)
                if result.is_complete:
                    completed = session.metadata.scale_state
                    if completed and completed.get("status") == "completed":
                        session.metadata.scale_history.append({
                            "scale_name": completed.get("scale_name"),
                            "total_score": completed.get("total_score"),
                            "level": completed.get("level"),
                            "escalation_flag": completed.get("escalation_flag", False),
                        })
                        if completed:
                            try:
                                from core.memory.db_storage import DatabaseStorage
                                DatabaseStorage.save_scale_screening(
                                    session_id=req.session_id,
                                    scale_type=completed.get("scale_name", "unknown"),
                                    state=completed.get("status", "completed"),
                                    scores=completed.get("scores"),
                                    total_score=float(completed.get("total_score", 0)) if completed.get("total_score") is not None else None,
                                    user_id=req.user_id or "",
                                )
                            except Exception:
                                pass

                        remaining = (completed or {}).get("pending_scales", [])
                        rejection_kw = ["不要", "先不谈", "太多了", "先不"]
                        should_abandon = any(kw in req.user_text for kw in rejection_kw)
                        if remaining and not should_abandon:
                            next_scale = remaining[0]
                            next_pending = remaining[1:]
                            next_reply = orch.start(next_scale, session)
                            if next_pending:
                                ns = session.metadata.scale_state
                                if ns:
                                    ns["pending_scales"] = next_pending
                                    session.metadata.scale_state = ns
                            self._get_generator()._save_turn(req.session_id, req.user_text, result.reply + "\n\n" + next_reply)
                            yield result.reply + "\n\n" + next_reply
                            return
                        elif remaining and should_abandon:
                            completed["pending_scales"] = []
                            session.metadata.scale_state = completed
                            for sn in remaining:
                                session.metadata.scale_history.append({
                                    "scale_name": sn,
                                    "status": "abandoned",
                                })
                                try:
                                    from core.memory.db_storage import DatabaseStorage
                                    DatabaseStorage.save_scale_screening(
                                        session_id=req.session_id,
                                        scale_type=sn,
                                        state="abandoned",
                                        user_id=req.user_id or "",
                                    )
                                except Exception:
                                    pass

                # 保存本轮对话（量表进行中也需持久化历史）
                self._get_generator()._save_turn(req.session_id, req.user_text, result.reply)
                yield result.reply  # 单 chunk，量表话术不用 LLM
                return

        # ── 医生模式 ──
        route, scid_enriched_query, scid_directive = self._run_doctor_assessment(
            req, session, route
        )

        # INTERVENTION TRACE
        logger.info("[PIPELINE:TRACE] INTERVENTION dispatch (stream) → %s", route)

        if route == "crisis":
            result = self._get_crisis().handle(req)
            self._save_crisis_turn(req.session_id, req.user_text, result.reply)
            yield result.reply  # 模板话术，单 chunk
            return

        if route == "general":
            async for token in self._get_generator().astream_general(
                req, scid_directive=scid_directive
            ):
                yield token
            return

        if route == "comfort":
            async for token in self._get_generator().astream_comfort(
                req, scid_directive=scid_directive
            ):
                yield token
            return

        if route == "knowledge":
            session = self._get_session(req.session_id)
            orch = self._get_orchestrator()
            # DOCTOR_MODE：跳过量表自动触发，走周医生自然对话
            if session is not None and orch is not None and not settings.DOCTOR_MODE:
                triggered_scales = orch.should_trigger(req.user_text, req.emotion or {})
                if triggered_scales:
                    scale_name = triggered_scales[0]
                    reply = orch.start(scale_name, session)
                    try:
                        from core.memory.db_storage import DatabaseStorage
                        DatabaseStorage.save_scale_screening(
                            session_id=req.session_id,
                            scale_type=scale_name,
                            state="in_progress",
                            user_id=req.user_id or "",
                        )
                    except Exception:
                        pass
                    pending = triggered_scales[1:]
                    if pending:
                        state_raw = session.metadata.scale_state
                        if state_raw:
                            state_raw["pending_scales"] = pending
                            session.metadata.scale_state = state_raw
                    # 保存本轮对话（量表路径也需持久化历史）
                    self._get_generator()._save_turn(req.session_id, req.user_text, reply)
                    self._get_generator()._save_probed_dimension(req.session_id, reply)
                    yield reply  # 量表启动话术，单 chunk
                    return

            if scid_enriched_query:
                enriched_query = scid_enriched_query
            else:
                enriched_query = req.user_text
                if session is not None:
                    from modules.intervention.rag.scale_feedback import enrich_query_with_scale
                    enriched_query = enrich_query_with_scale(req.user_text, session)
            async for token in self._get_generator().astream_knowledge(
                req, enriched_query=enriched_query, scid_directive=scid_directive
            ):
                yield token
            return

        yield "[placeholder] 待实现"
