"""REMSPipeline：匠石记忆后端（REMS3 fork）新编排。

- **记忆写入**：``ingest_batch``（MemoryBackendPort）——逐 input 代谢封存，写对象时间线 /
  stored_marks / 向量索引（design/210、610、810、1010）；
- **回忆读取**：``recall``（MemoryBackendPort）——RecallPipeline 多路召回（design/1010）；
- **信念修正**：``tombstone``——保留墓碑标记，只降可见性、不删原文（design/10）。
"""

from __future__ import annotations

import logging

from .config import REMSConfig
from .llm.provider import LLMProvider
from .models.event import Event
from .port import BackendIngestResult, MemoryBatch, RecalledFragment
from .recall import (
    BgeReranker,
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallPipeline,
    SentenceTransformerEmbedding,
)
from .services.belief_revision_service import BeliefRevisionService
from .services.emotion_service import EMAEvolver
from .services.event_service import EventService
from .services.metabolism_service import MetabolismService
from .services.role_service import RoleService
from .skills.boundary_detection import BoundaryDetectionSkill
from .skills.event_enrichment import EventEnrichmentSkill
from .skills.role_extraction import RoleExtractionSkill
from .storage.database import Database
from .storage.repository import (
    EventRepository,
    MetabolismRepository,
    ObjectTimelineRepository,
    RoleRepository,
    StoredMarksRepository,
)
from .strategies.event_weight import EventWeightDeriver

logger = logging.getLogger(__name__)


class REMSPipeline:
    """记忆写入 + 回忆读取双端口编排。"""

    def __init__(
        self,
        config: REMSConfig,
        llm: LLMProvider,
        db: Database,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        meta_repo: MetabolismRepository,
        event_service: EventService,
        role_service: RoleService,
        metabolism_service: MetabolismService,
        belief_revision_service: BeliefRevisionService,
        *,
        recall_pipeline: RecallPipeline | None = None,
        object_timeline_repo: ObjectTimelineRepository | None = None,
        stored_marks_repo: StoredMarksRepository | None = None,
    ):
        self.config = config
        self.llm = llm
        self.db = db
        self.event_repo = event_repo
        self.role_repo = role_repo
        self.meta_repo = meta_repo
        self.event_service = event_service
        self.role_service = role_service
        self.metabolism_service = metabolism_service
        self.belief_revision_service = belief_revision_service
        self.recall_pipeline = recall_pipeline
        self.object_timeline_repo = object_timeline_repo
        self.stored_marks_repo = stored_marks_repo

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: REMSConfig | None = None) -> "REMSPipeline":
        """工厂：装配存储、技能、领域服务与回忆 provider（插拔式，design/1010 §9.1）。"""
        if config is None:
            config = REMSConfig()

        llm = LLMProvider(config)
        db = Database(config.storage.database_url)
        db.create_tables()

        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        meta_repo = MetabolismRepository(db)
        object_timeline_repo = ObjectTimelineRepository(db)
        stored_marks_repo = StoredMarksRepository(db)

        # ---- 回忆 provider（插拔式） ----
        if config.embedding.provider == "hash":
            embedding = HashEmbedding()
        else:
            embedding = SentenceTransformerEmbedding(config.embedding.model_name)
        vector_store = QdrantRecallVectorStore(
            config.storage.qdrant_collection,
            url=config.storage.qdrant_url,
            path=config.storage.qdrant_path,
        )
        # reranker 默认 Null（纯 RRF）；启用时换 BgeReranker（见配置项，后续接入）。
        reranker: BgeReranker | NullReranker = NullReranker()
        recall_pipeline = RecallPipeline(
            embedding,
            vector_store,
            event_repo,
            object_timeline_repo,
            reranker=reranker,
        )

        # ---- 技能与领域服务 ----
        role_skill = RoleExtractionSkill(llm, config)
        boundary_skill = BoundaryDetectionSkill(llm, config)
        enrichment_skill = EventEnrichmentSkill(llm, config, role_fallback=role_skill)

        event_weight = EventWeightDeriver(config, event_repo, role_repo)
        emotion_evolver = EMAEvolver(config, role_repo)
        role_service = RoleService(config, role_repo, role_skill, llm=llm)

        event_service = EventService(
            config,
            llm,
            event_repo,
            None,  # 旧向量库不再使用（design/810：向量占位 → 回忆由 recall_pipeline 索引）
            enrichment_skill,
            role_service=role_service,
            emotion_evolver=emotion_evolver,
            event_weight=event_weight,
            object_timeline_repo=object_timeline_repo,
        )
        metabolism_service = MetabolismService.with_default_boundary_repair(
            config,
            meta_repo,
            boundary_skill,
            event_service,
            event_repo=event_repo,
            llm=llm,
        )
        belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

        return cls(
            config=config,
            llm=llm,
            db=db,
            event_repo=event_repo,
            role_repo=role_repo,
            meta_repo=meta_repo,
            event_service=event_service,
            role_service=role_service,
            metabolism_service=metabolism_service,
            belief_revision_service=belief_revision_service,
            recall_pipeline=recall_pipeline,
            object_timeline_repo=object_timeline_repo,
            stored_marks_repo=stored_marks_repo,
        )

    # ------------------------------------------------------------------
    # 记忆写入端口（design/210/610）
    # ------------------------------------------------------------------

    def ingest_batch(self, batch: MemoryBatch) -> BackendIngestResult:
        """记忆写入：逐 input 走代谢封存，写对象时间线 / stored_marks / 向量索引。

        单条失败进 ``errors``，不污染其他 input 与历史；``stored_marks`` 封存几个填几个。
        """
        result = BackendIngestResult(subject_id=batch.subject_id)
        for inp in batch.experiences:
            if not (inp.text or "").strip():
                result.errors.append(f"segment {inp.segment_id or '?'}: empty text")
                continue
            try:
                sealed = self.metabolism_service.process_input(inp.text, memory=inp)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ingest_batch segment %s failed: %s", inp.segment_id, exc)
                result.errors.append(f"segment {inp.segment_id}: {exc}")
                continue
            for ev in sealed:
                result.sealed_event_ids.append(ev.event_id)
                for re_ in ev.role_list:
                    if (
                        not re_.is_subject
                        and re_.role_id
                        and re_.role_id not in result.role_ids
                    ):
                        result.role_ids.append(re_.role_id)
                if self.recall_pipeline is not None:
                    try:
                        self.recall_pipeline.index_event(ev)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("index_event %s failed: %s", ev.event_id, exc)
                        result.errors.append(f"index {ev.event_id}: {exc}")
            if sealed and inp.segment_id:
                ids = [e.event_id for e in sealed]
                result.stored_marks[inp.segment_id] = ids
                if self.stored_marks_repo is not None:
                    try:
                        self.stored_marks_repo.save(batch.subject_id, inp.segment_id, ids)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("stored_marks save failed for %s: %s", inp.segment_id, exc)
                        result.errors.append(f"stored_marks {inp.segment_id}: {exc}")
        result.unclosed_count = len(self.meta_repo.get_unclosed_events())
        return result

    # ------------------------------------------------------------------
    # 回忆端口（design/1010）
    # ------------------------------------------------------------------

    def recall(
        self,
        subject_id: str,
        query: str,
        *,
        object_id: str | None = None,
        level: int = 1,
        limit: int | None = None,
        anchor_event_ids: tuple[str, ...] = (),
    ) -> tuple[RecalledFragment, ...]:
        """实现 MemoryBackendPort.recall（只读 + 记忆恢复）。"""
        if self.recall_pipeline is None:
            return ()
        return self.recall_pipeline.recall(
            subject_id,
            query,
            object_id=object_id,
            level=level,
            limit=limit,
            anchor_event_ids=anchor_event_ids,
        )

    # ------------------------------------------------------------------
    # 便捷方法（API 用）
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

    def tombstone(self, event_id: str, reason: str, replacement_id: str | None = None) -> bool:
        return self.belief_revision_service.tombstone_event(
            event_id, reason=reason, replacement_event_id=replacement_id
        )
