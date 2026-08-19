from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .config import REMSConfig, UserMode
from .llm.ingest_session import IngestLlmSession
from .llm.provider import LLMProvider
from .models.event import Event, EventRoleEntry
from .models.metabolism import ContextPackage
from .models.role import Role
from .observability import PerfMonitor
from .observability.ingest_trace import IngestTrace, event_preview
from .services.abstraction_service import AbstractionService
from .services.belief_revision_service import BeliefRevisionService
from .services.emotion_service import EMAEvolver
from .services.event_service import EventService
from .services.metabolism_service import MetabolismService
from .services.recall_quality import RecallQualityController
from .retrieval import EventBm25Index, apply_profile_defaults, create_recall_backend
from .services.recall_service import RecallService
from .services.role_service import RoleService
from .skills.boundary_detection import BoundaryDetectionSkill
from .skills.event_enrichment import EventEnrichmentSkill
from .skills.inductive_evolution import InductiveEvolutionSkill
from .skills.recall_block_relevance import RecallBlockRelevanceSkill
from .skills.recall_query_compress import RecallQueryCompressSkill
from .skills.role_extraction import RoleExtractionSkill
from .skills.summary_generation import SummaryGenerationSkill
from .storage.database import Database
from .storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    MetabolismRepository,
    RecallLogRepository,
    RoleRepository,
)
from .embedding.tri_band import TriBandEncoder
from .embedding.query_act_processor import build_query_act_processor
from .ingest.focus_roles import match_focus_role_ids
from .services.abstraction_shadowing import AbstractionShadowingService
from .skills.recall_intent import RecallIntentClassifier
from .storage.active_pool_cache import ActivePoolCache
from .storage.tier1_store import Tier1Store
from .storage.vector_store import VectorStore
from .strategies.event_weight import EventWeightDeriver

# 顶层编排：ingest 串联回忆（ContextPackage）、代谢封存、角色白描/语义卡片、
# 记忆再巩固抽象与 NPC 指令；对照《REMS 记忆系统规范解析》4.4、5.1–5.3。

logger = logging.getLogger(__name__)


# =====================================================================
# Processing Modes (§5 Multi-scenario Output)
# =====================================================================

class ProcessingMode(str, Enum):
    """Output mode controlling what the pipeline **returns** to the caller.

    重要设计约定（与白皮书 §5 一致）：**模式只影响"输出形态"，不影响"内部记忆动力学"**。
    回忆块组装、``recall_log`` 登记、频繁子集挖掘 (§3.2) 等一律无条件执行——因为：
        (a) 抽象事件的唯一触发路径是 ``recall_log``，跳过回忆等于放弃所有抽象合成；
        (b) 白描、AE、语义卡片等后台巩固机制都依赖对事件的持续检索相关性；
        (c) "静默倾听"的语义是「不对用户可见」，而不是「不要构建记忆」。

    所以模式的差异收敛到一个布尔属性 :pyattr:`returns_context_package` 以及少量输出
    装配逻辑（例如 NPC 的 Directive）。新增模式时只需在枚举中追加成员、覆盖属性或
    扩展 ``_build_*_directives``，不需要触动 ``ingest`` 的主干流程。

    DIALOGUE
        标准强输出交互。ingest 返回完整 ContextPackage，供上游 LLM 生成自然语言回复。
    PASSIVE_LOG
        "静默倾听"：穿戴录音、会议转写等「只记不说」场景。回忆、代谢、抽象合成**仍然执行**
        且登记入库，只是 ``ContextPackage`` 不对外暴露（:pyattr:`returns_context_package`
        ``= False``），不产出人类可见的文本回复。
    NPC_AGENT
        生成式 NPC / 沙盒 Agent。除了 ContextPackage 外，还输出结构化 ``npc_directives``
        （动作 + 情绪增量等），交给下游 Directive Parser 转为引擎调用。
    """
    DIALOGUE = "dialogue"
    PASSIVE_LOG = "passive_log"
    NPC_AGENT = "npc_agent"

    @property
    def returns_context_package(self) -> bool:
        """Whether ``ingest`` should expose the ContextPackage to the caller.

        默认对外暴露；被动日志等"静默"模式重写为 False。新增对外不回复、只沉淀记忆的模式
        时复写此属性即可，内部回忆与抽象管线不需要改动。
        """
        return self != ProcessingMode.PASSIVE_LOG


# =====================================================================
# Result dataclasses
# =====================================================================

@dataclass
class ProcessingResult:
    """Outcome of a single ``ingest`` call across processing modes.

    sealed_events — newly sealed basic events from this metabolism pass.
    abstract_events — abstract events synthesised via reconsolidation or post-seal clustering.
    context_package — recall + shadow + current input (omitted in PASSIVE_LOG).
    npc_directives — structured NPC actions when ``mode == NPC_AGENT``.

    单次 ``ingest`` 的返回结果（多场景共用同一结构）：
    ``sealed_events`` 为本轮代谢新封存的基本事件；
    ``abstract_events`` 为记忆再巩固或封存后聚类产生的抽象事件；
    ``context_package`` 为回忆块+残影+当前输入（被动日志模式下为 None）；
    ``npc_directives`` 为 NPC 模式下的结构化行为指令列表。
    """
    # 一次 ingest 的产出：新封存基本事件、（可选）新抽象事件、对话用上下文包、NPC 指令。
    sealed_events: list[Event] = field(default_factory=list)
    abstract_events: list[Event] = field(default_factory=list)
    context_package: Optional[ContextPackage] = None
    mode: ProcessingMode = ProcessingMode.DIALOGUE

    # NPC 模式下的结构化行为输出（白皮书 5.3）。
    npc_directives: list[dict] = field(default_factory=list)


# =====================================================================
# Pipeline
# =====================================================================

class REMSPipeline:
    """Top-level orchestrator wiring all REMS components together.

    Usage::

        pipeline = REMSPipeline.from_config(config)

        # Interactive dialogue
        result = pipeline.ingest("今天发生了一件事……")

        # Passive logging (silent, no ctx package)
        result = pipeline.ingest("…", mode=ProcessingMode.PASSIVE_LOG)

        # NPC agent
        result = pipeline.ingest("玩家靠近", mode=ProcessingMode.NPC_AGENT,
                                 npc_role_id="ROL-xxx")

    顶层编排器：将配置、LLM、持久化、向量库、各仓储与领域服务串联为可调用管线。
    典型用法见上；``from_config`` 负责一次性构造全部依赖（技能与服务的依赖注入入口）。
    """

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        db: Database,
        vector_store: VectorStore,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        meta_repo: MetabolismRepository,
        recall_log_repo: RecallLogRepository,
        event_service: EventService,
        role_service: RoleService,
        metabolism_service: MetabolismService,
        recall_service: RecallService,
        abstraction_service: AbstractionService,
        belief_revision_service: BeliefRevisionService,
        role_skill: RoleExtractionSkill | None = None,
        perf_monitor: PerfMonitor | None = None,
        recall_quality: RecallQualityController | None = None,
        recall_block_relevance_skill: RecallBlockRelevanceSkill | None = None,
        ingest_session: IngestLlmSession | None = None,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.vector_store = vector_store
        self.event_repo = event_repo
        self.role_repo = role_repo
        self.meta_repo = meta_repo
        self.recall_log_repo = recall_log_repo
        self.event_service = event_service
        self.role_service = role_service
        self.metabolism_service = metabolism_service
        self.recall_service = recall_service
        self.abstraction_service = abstraction_service
        self.belief_revision_service = belief_revision_service
        self.role_skill = role_skill
        # 暴露 perf_monitor，便于上层脚本读取耗时指标 / 当前 load_factor 做诊断。
        self.perf_monitor = perf_monitor
        self.recall_quality = recall_quality
        self.recall_block_relevance_skill = recall_block_relevance_skill
        self._ingest_session = ingest_session
        self._ingest_seq = 0
        self.ingest_trace: IngestTrace | None = None
        # 单线程后台执行器：把回忆前的人物提取（纯 LLM，不碰 DB）与回忆重叠。
        # max_workers=1 足够——每轮 ingest 最多并发一次人物提取；DB 登记仍在主线程串行完成。
        self._role_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rems-role"
        )

    def _begin_ingest_step(self, step: str) -> float:
        trace = self.ingest_trace
        if trace is not None:
            trace.set_current_step(step)
        return time.perf_counter()

    def _trace_ingest_step(self, step: str, t0: float, **detail: Any) -> None:
        trace = self.ingest_trace
        if trace is None:
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        trace.record(step, elapsed_ms, **detail)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: REMSConfig | None = None) -> REMSPipeline:
        # Factory: wire storage, skills, and domain services in one place for API/scripts.
        # 工厂方法：集中实例化存储、技能与领域服务，供 API 与脚本共用。
        if config is None:
            config = REMSConfig()

        apply_profile_defaults(config, config.recall_profile)

        llm = LLMProvider(config)
        db = Database(config.storage.database_url)
        db.create_tables()

        # 性能监控（白皮书 §2.3 高负载自我保护）：
        # 监听 RAG / 回忆组装 / 抽象挖掘 / 叙事判重 等非 LLM 主算法的滚动均耗时，
        # load_factor 上升时驱动 (1) 白描静默阈值收紧 (2) 抽象判重 K 收窄。
        perf_monitor = PerfMonitor(
            enabled=config.perf_monitor_enabled,
            window_size=config.perf_monitor_window_size,
            tolerances_ms=config.perf_phase_tolerance_ms,
            load_factor_max=config.perf_load_factor_max,
            recall_phases=config.perf_recall_phases,
            abstract_phases=config.perf_abstract_phases,
            silence_boost_at_max=config.forgetting_overload_silence_boost,
        )

        tier1_store = Tier1Store(db, config)
        active_pool_cache = ActivePoolCache(tier1_store, config)
        active_pool_cache.refresh()

        vector_store = VectorStore(config, perf_monitor=perf_monitor)
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
        vector_store.set_tri_band(tri_band)
        event_weight = EventWeightDeriver(config, event_repo, role_repo)
        asf_service = AbstractionShadowingService(
            config, event_repo, role_repo, vector_store, tier1_store,
        )
        meta_repo = MetabolismRepository(db)
        recall_log_repo = RecallLogRepository(db)
        abstracted_subset_repo = AbstractedSubsetRepository(db)

        recall_quality = RecallQualityController(config)

        # SummaryGenerationSkill 仍用于抽象事件（inductive evolution 后的总结）。
        # 基本事件流已被 EventEnrichmentSkill 接管（一次调用产出摘要 + 角色）。
        summary_skill = SummaryGenerationSkill(llm, config)
        role_skill = RoleExtractionSkill(llm, config)
        boundary_skill = BoundaryDetectionSkill(llm, config)
        enrichment_skill = EventEnrichmentSkill(llm, config, role_fallback=role_skill)
        evolution_skill = InductiveEvolutionSkill(llm, config)
        relevance_skill = RecallBlockRelevanceSkill(llm, config)
        query_compress_skill = RecallQueryCompressSkill(llm, config)
        query_act_processor = build_query_act_processor(config, skill=query_compress_skill)

        emotion_evolver = EMAEvolver(config, role_repo)
        # Pass llm to role_service so semantic cards can be refreshed in-process
        role_service = RoleService(config, role_repo, role_skill, llm=llm, vector_store=vector_store)

        bm25_index = EventBm25Index()
        bm25_index.set_role_repo(role_repo)
        bm25_index.rebuild(
            [
                e
                for e in event_repo.list_all(exclude_tombstoned=True)
                if e.status.value != "silent"
            ]
        )

        event_service = EventService(
            config,
            llm,
            event_repo,
            vector_store,
            enrichment_skill,
            role_service=role_service,
            emotion_evolver=emotion_evolver,
            tier1_store=tier1_store,
            event_weight=event_weight,
            bm25_index=bm25_index,
        )
        
        metabolism_service = MetabolismService.with_default_boundary_repair(
            config,
            meta_repo,
            boundary_skill,
            event_service,
            event_repo=event_repo,
            llm=llm,
        )
        recall_backend = create_recall_backend(
            config, event_repo, role_repo, bm25_index,
        )
        recall_service = RecallService(
            config, event_repo, role_repo, vector_store, perf_monitor=perf_monitor,
            recall_quality=recall_quality,
            active_pool_cache=active_pool_cache,
            tri_band=tri_band,
            event_weight=event_weight,
            intent_classifier=RecallIntentClassifier(config, embedder=tri_band),
            query_act_processor=query_act_processor,
            recall_backend=recall_backend,
            bm25_index=bm25_index,
        )
        abstraction_service = AbstractionService(
            config,
            event_repo,
            vector_store,
            evolution_skill,
            summary_skill,
            recall_log_repo,
            abstracted_subset_repo,
            enrichment_skill=enrichment_skill,
            perf_monitor=perf_monitor,
            recall_quality=recall_quality,
            asf_service=asf_service,
        )
        belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

        ingest_session = IngestLlmSession(llm, config, boundary_skill)

        return cls(
            config=config,
            llm=llm,
            db=db,
            vector_store=vector_store,
            event_repo=event_repo,
            role_repo=role_repo,
            meta_repo=meta_repo,
            recall_log_repo=recall_log_repo,
            event_service=event_service,
            role_service=role_service,
            metabolism_service=metabolism_service,
            recall_service=recall_service,
            abstraction_service=abstraction_service,
            belief_revision_service=belief_revision_service,
            role_skill=role_skill,
            perf_monitor=perf_monitor,
            recall_quality=recall_quality,
            recall_block_relevance_skill=relevance_skill,
            ingest_session=ingest_session,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def ingest(
        self,
        raw_input: str,
        *,
        force_save: bool = False,
        mode: ProcessingMode = ProcessingMode.DIALOGUE,
        npc_role_id: str | None = None,
        input_id: str | None = None,
        ingest_seq: int | None = None,
    ) -> ProcessingResult:
        """Full processing cycle.

        **内部记忆动力学（所有模式一律执行）**：
            1. 取残影 + 焦点角色；
            2. 组装 ``ContextPackage``（回忆块 + 残影 + 当前输入）；
            3. 把回忆块真实事件 ID 登记到 ``recall_log``——白皮书 §3.2 规定这是抽象事件的**唯一**触发路径，
               因此即使被动日志模式"不对用户说话"，也必须完成回忆与登记，否则系统永远不会演化出抽象规律；
            4. 代谢：边界检测、封存基本事件、维护残影与未完成库（§4.1–§4.2）；
            5. 角色更新：对每个新基本事件追加白描时间线、必要时刷新语义卡片（§2.2–§2.3，抽象事件自动跳过）；
            6. 抽象合成：扫 ``recall_log`` 全量历史做频繁子集挖掘（§3.2）。

        **外部输出形态（由 ``mode`` 决定）**：
            - ``returns_context_package == True``：对外返回 ContextPackage 供上游 LLM 生成回复；
            - ``returns_context_package == False``：ContextPackage 留在内部，对外返回 ``None``
              （"静默倾听"）；
            - ``NPC_AGENT``：额外基于回忆块派生 ``npc_directives``；
            - 新增模式只需在 :class:`ProcessingMode` 追加成员并复写 ``returns_context_package``
              （或扩展专属装配步骤），不需要改动主干流程。
        """
        t_ingest = time.perf_counter()

        t0 = self._begin_ingest_step("shadow_load")
        shadow = self.meta_repo.get_shadow()
        unclosed_before = self.meta_repo.get_unclosed_events()
        self._trace_ingest_step(
            "shadow_load",
            t0,
            shadow_chars=len(shadow.content or ""),
            unclosed_count=len(unclosed_before),
        )

        # Step 1: Turn1 检索 query 压缩（阻塞 recall）；Turn2 角色与 recall 并发。
        combined_text = (shadow.content + "\n" + raw_input).strip()
        use_session = (
            self.config.ingest_llm_session_enabled
            and self._ingest_session is not None
            and bool(combined_text)
        )
        logger.debug(
            "ingest pre-recall: session=%s, combined_text_len=%d, wait=%s",
            use_session,
            len(combined_text),
            self.config.recall_wait_for_role_extraction,
        )

        act_source: str | None = None
        act_source_label = "llm_compress"
        role_future: Future | None = None
        role_entries: list[EventRoleEntry] = []
        extracted_names: list[str] = []
        role_extract_waited = False

        if use_session:
            self._ingest_session.reset()
            t_compress = self._begin_ingest_step("recall_query_compress")
            compress_skipped = False
            try:
                act_source = self._ingest_session.compress_search_text(combined_text)
                compress_skipped = (
                    not self.config.recall_query_compress_enabled
                    or (
                        len(combined_text) <= self.config.recall_query_compress_min_chars
                        and len(combined_text) <= self.config.tri_band.query_act_compress_max_chars
                    )
                )
            except Exception as exc:
                logger.warning("ingest session Turn1 compress failed: %s", exc)
                act_source = None
            act_preview = (act_source or "")[:120]
            if act_preview and len(act_source or "") > 120:
                act_preview += "…"
            self._trace_ingest_step(
                "recall_query_compress",
                t_compress,
                act_chars=len(act_source or ""),
                act_query_preview=act_preview or None,
                compress_skipped=compress_skipped,
                compress_source="RecallQueryCompressSkill",
                combined_text_chars=len(combined_text),
                session_turn=self._ingest_session.turn,
            )

        t0 = self._begin_ingest_step("pre_recall_role_extract")
        if use_session:
            role_future = self._role_executor.submit(
                self._ingest_session.extract_roles_followup,
                None,
            )
        elif self.role_skill and combined_text:
            role_future = self._role_executor.submit(self.role_skill.extract, combined_text)

        if role_future is not None and self.config.recall_wait_for_role_extraction:
            role_entries, extracted_names = self._consume_role_future(role_future)
            role_future = None
            role_extract_waited = True

        # 焦点角色：已登记的抽取结果（若已等待）+ 单人核心用户 + 即时启发式名匹配。
        focus_role_ids = set(r.role_id for r in role_entries)
        if self.config.user_mode == UserMode.SINGLE and self.config.core_user_role_id:
            focus_role_ids.add(self.config.core_user_role_id)
        focus_role_ids.update(
            match_focus_role_ids(
                self.role_repo,
                raw_input=raw_input,
                shadow_content=shadow.content if shadow else "",
                extra_texts=[act_source] if act_source else (),
            )
        )
        self._trace_ingest_step(
            "pre_recall_role_extract",
            t0,
            combined_text_chars=len(combined_text),
            session=use_session,
            submitted=role_future is not None or role_extract_waited,
            waited=role_extract_waited,
            extracted_roles=len(role_entries),
            role_ids=sorted({r.role_id for r in role_entries}),
            role_names=extracted_names[:12],
            focus_role_ids=sorted(focus_role_ids),
            act_source_precomputed=act_source is not None,
        )

        # 回忆（所有模式必须执行）：
        t0 = self._begin_ingest_step("recall_build_context")
        recall_profile = self.recall_service.refresh_backend(mode.value)
        recall_kwargs: dict[str, Any] = {
            "focus_role_ids": focus_role_ids,
            "focus_role_entries": role_entries,
        }
        if act_source is not None:
            recall_kwargs["act_source"] = act_source
            recall_kwargs["act_source_label"] = act_source_label
        if ingest_seq is not None:
            recall_kwargs["ingest_seq"] = ingest_seq
        ctx: ContextPackage = self.recall_service.build_context_package(
            raw_input, shadow, **recall_kwargs,
        )
        recall_block_ids = [
            it.event_id for it in (ctx.recall_block.items or []) if getattr(it, "event_id", None)
        ]
        self._trace_ingest_step(
            "recall_build_context",
            t0,
            recall_items=len(ctx.recall_block.items or []),
            event_ids=recall_block_ids[:12],
            shadow_in_ctx=len(ctx.shadow.content or "") if ctx.shadow else 0,
            recall_profile=recall_profile,
        )

        # 登记本次回忆块 event_id 到 recall_log（白皮书 §3.2 唯一抽象触发路径的输入流）。
        # 只记录真实 basic/abstract 事件；抽象事件 id 同样进入命名空间，
        # 便于后续更高阶抽象在同一空间继续挖掘。
        t0 = self._begin_ingest_step("recall_log_append")
        recall_event_ids: list[str] = []
        recall_id: str | None = None
        if ctx.recall_block.items:
            seen_ids: set[str] = set()
            for it in ctx.recall_block.items:
                eid = it.event_id
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                recall_event_ids.append(eid)
            if recall_event_ids:
                recall_id = f"RCL-{uuid.uuid4().hex}"
                self.recall_log_repo.append(
                    recall_id=recall_id,
                    event_ids=recall_event_ids,
                )
        self._trace_ingest_step(
            "recall_log_append",
            t0,
            appended=bool(recall_event_ids),
            recall_id=recall_id,
            n_events=len(recall_event_ids),
            event_ids=recall_event_ids[:12],
        )

        self._ingest_seq += 1
        t0 = self._begin_ingest_step("recall_relevance_audit")
        rq, rel_skill = self.recall_quality, self.recall_block_relevance_skill
        audited = False
        relevance_rate: float | None = None
        if (
            rq is not None
            and rel_skill is not None
            and rq.should_audit_relevance(self._ingest_seq)
            and ctx.recall_block.items
        ):
            try:
                relevance_rate = rel_skill.evaluate_relevance_ratio(raw_input, ctx.recall_block)
                rq.record_recall_block_relevance_rate(relevance_rate)
                audited = True
            except Exception as exc:
                logger.warning("recall block relevance audit failed: %s", exc)
        self._trace_ingest_step(
            "recall_relevance_audit",
            t0,
            ran=audited,
            relevance_rate=relevance_rate,
        )

        # 进代谢前取回并发的人物提取结果（若回忆阶段未等待）。抽取与回忆已重叠，
        # 此处 await 通常接近零；在主线程完成 resolve_and_register（避免与回忆并发写 DB）。
        if role_future is not None:
            t0 = self._begin_ingest_step("role_extract_await")
            role_entries, extracted_names = self._consume_role_future(role_future)
            role_future = None
            self._trace_ingest_step(
                "role_extract_await",
                t0,
                extracted_roles=len(role_entries),
                role_ids=sorted({r.role_id for r in role_entries}),
                role_names=extracted_names[:12],
            )

        # 代谢：边界检测、封存基本事件、维护残影与未完成库（第 4.1–4.2）。
        # 把 pre-recall 已提取的角色作为 enrichment 去代词化 hint 透传，避免事件层面重复抽取。
        # 同时把抽到的富信息 EventRoleEntry（含 snapshot + 8 维情绪）作为「全局池」下放：
        # EventService.seal_event 会让 enrichment 走 names_only 分支，仅识别本事件登场的角色名，
        # 然后按 role_id 从池中回填 snapshot/情感，避免事件级别重复 LLM 抽取（白皮书 2.1）。
        known_roles_hint: list[Role] = []
        if role_entries:
            for re_entry in role_entries:
                role_obj = self.role_repo.get(re_entry.role_id)
                if role_obj is not None:
                    known_roles_hint.append(role_obj)
        t0 = self._begin_ingest_step("metabolism_seal")
        sealed = self.metabolism_service.process_input(
            raw_input,
            force_save=force_save,
            input_id=input_id,
            known_roles_hint=known_roles_hint or None,
            pre_role_entries=role_entries or None,
            ingest_session=self._ingest_session if use_session else None,
        )
        unclosed_after = self.meta_repo.get_unclosed_events()
        self._trace_ingest_step(
            "metabolism_seal",
            t0,
            sealed_count=len(sealed),
            sealed=[event_preview(e) for e in sealed],
            unclosed_after=len(unclosed_after),
            force_save=force_save,
        )

        # 角色：每个新事件更新白描时间线并刷新语义卡片（第 2.2–2.3）。
        # RoleService 对 is_abstract=True 的事件自带早退，抽象事件不会污染白描。
        t0 = self._begin_ingest_step("role_timeline_update")
        for event in sealed:
            self.role_service.update_from_event(event)
        self._trace_ingest_step(
            "role_timeline_update",
            t0,
            events_updated=len(sealed),
            role_ids=sorted({r.role_id for e in sealed for r in e.role_list}),
        )

        # 抽象事件触发 —— 唯一路径：``recall_log`` 中的频繁子集挖掘（白皮书 §3.2）。
        # 所有模式都跑：被动日志场景下仍然需要持续演化出抽象规律供未来检索或审计。
        t0 = self._begin_ingest_step("abstraction_mine")
        abstract_events: list[Event] = self.abstraction_service.mine_and_synthesize()
        self._trace_ingest_step(
            "abstraction_mine",
            t0,
            new_abstracts=len(abstract_events),
            abstracts=[event_preview(e) for e in abstract_events],
        )

        # NPC：由回忆与威胁启发式生成行为指令（第 5.3）。
        npc_directives: list[dict] = []
        if mode == ProcessingMode.NPC_AGENT and npc_role_id:
            npc_directives = self._build_npc_directives(npc_role_id, ctx, sealed)

        # 输出路由：``returns_context_package`` 控制 ctx 是否对外暴露；内部管线与产物不变。
        exposed_ctx: Optional[ContextPackage] = ctx if mode.returns_context_package else None

        self._begin_ingest_step("ingest_complete")
        self._trace_ingest_step(
            "ingest_complete",
            t_ingest,
            mode=mode.value,
            input_chars=len(raw_input),
            sealed_total=len(sealed),
            abstract_total=len(abstract_events),
        )

        return ProcessingResult(
            sealed_events=sealed,
            abstract_events=abstract_events,
            context_package=exposed_ctx,
            mode=mode,
            npc_directives=npc_directives,
        )

    # ------------------------------------------------------------------
    # Focus role extraction (for recall tier selection)
    # ------------------------------------------------------------------

    def _consume_role_future(
        self, role_future: Future
    ) -> tuple[list[EventRoleEntry], list[str]]:
        """Resolve the background role-extraction future and register roles on the main thread.

        线程内只做了纯 LLM 抽取；此处在主线程完成 ``resolve_and_register``（DB 读写）与
        ``EventRoleEntry`` 构造，保证 DB 不被回忆/抽取并发写入。任何异常都降级为空结果，
        让主流程继续（人物提取失败不应阻断回忆与代谢）。
        """
        role_entries: list[EventRoleEntry] = []
        extracted_names: list[str] = []
        try:
            extraction_result = role_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("background role extraction failed: %s", exc)
            return role_entries, extracted_names

        extracted_roles = list(extraction_result.roles)
        extracted_names = [er.name for er in extracted_roles if er.name]
        if extracted_roles:
            id_mapping = self.role_service.resolve_and_register(extracted_roles, is_suspicious=False)
            for er in extracted_roles:
                lookup_key = er.role_id or er.name
                assigned_id = id_mapping.get(lookup_key, lookup_key)
                role_entries.append(self.role_skill.to_event_role_entry(er, assigned_id))
        return role_entries, extracted_names

    def _extract_focus_roles(
        self,
        raw_input: str,
        shadow_content: str = "",
        *,
        extra_texts: tuple[str, ...] = (),
    ) -> set[str]:
        """Backward-compatible wrapper; prefer ``match_focus_role_ids``."""
        return match_focus_role_ids(
            self.role_repo,
            raw_input=raw_input,
            shadow_content=shadow_content,
            extra_texts=extra_texts,
        )

    # ------------------------------------------------------------------
    # NPC / Generative-Agent directives
    # ------------------------------------------------------------------

    def _build_npc_directives(
        self,
        npc_role_id: str,
        ctx: Optional[ContextPackage],
        sealed: list[Event],
    ) -> list[dict]:
        """Derive structured action directives for an NPC role.

        Looks for memory of player interactions stored in the role's
        white-painting and the current recall block, then outputs a simple
        action + emotion-delta dict.  The downstream Directive Parser
        translates this to engine calls.

        为指定 ``npc_role_id`` 生成结构化 NPC 指令：在当前 ``ContextPackage`` 的回忆条目中，
        用启发式统计该 ID 在文本中的出现次数以估计威胁程度，并映射为 ``idle`` / ``watch`` /
        ``flee`` 等 ``action`` 及 ``emotion_delta``。下游 Directive Parser 将其翻译为
        寻路、动画或状态机变更（白皮书 5.3）；本实现为轻量示例，非完整游戏 AI。
        """
        directives: list[dict] = []
        if not ctx:
            return directives

        # Gather anger/threat signals from recall
        threat_score = 0.0
        for item in ctx.recall_block.items:
            if npc_role_id in item.content:
                threat_score += 0.2  # heuristic bump per mention

        action = "idle"
        emotion_delta: dict[str, float] = {}
        if threat_score >= 0.6:
            action = "flee"
            emotion_delta = {"fear": 0.4, "anger": 0.2}
        elif threat_score >= 0.3:
            action = "watch"
            emotion_delta = {"anticipation": 0.2, "fear": 0.1}

        if action != "idle":
            directives.append({
                "npc_role_id": npc_role_id,
                "action": action,
                "emotion_delta": emotion_delta,
                "reason": f"threat_score={threat_score:.2f}",
            })
        return directives

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def query_role(self, name_or_id: str) -> dict:
        role = self.role_repo.get(name_or_id) or self.role_service.find_role(name_or_id)
        if not role:
            return {"error": f"Role '{name_or_id}' not found"}
        summary = self.role_service.get_white_painting_summary(role.role_id)
        return {
            "role": role.model_dump(mode="json", exclude={"white_painting"}),
            "white_painting_summary": summary,
        }

    def run_evolution(self) -> list[Event]:
        """Manually drive the subset-mining pass (白皮书 §3.2).

        Equivalent to what ``ingest`` already runs after each recall; exposed for batch jobs.
        """
        return self.abstraction_service.mine_and_synthesize()

    def resolve_to_basic_events(self, event_id: str) -> list[Event]:
        """Resolve an abstract event to the flat list of basic events it ultimately derives from.

        从某条事件出发，沿 ``source_events`` 的嵌套链下钻直到叶子层（基本事件），
        返回 ``Event`` 模型列表（白皮书 §1.1.8 / §3.2）。
        输入若是基本事件，直接返回含自身的单元素列表；抽象事件会跨多层抽象逐级展开；
        墓碑事件不会进入结果。
        """
        basic_ids = self.event_repo.resolve_basic_event_ids(event_id)
        events: list[Event] = []
        for eid in basic_ids:
            ev = self.event_repo.get(eid)
            if ev is not None:
                events.append(ev)
        return events

    def tombstone(self, event_id: str, reason: str, replacement_id: str | None = None) -> bool:
        return self.belief_revision_service.tombstone_event(
            event_id, reason=reason, replacement_event_id=replacement_id
        )
