from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

# SQLAlchemy ORM：事件、角色、白描条目、残影、未完成事件的持久化表定义与 Database 门面。
# 领域含义见《REMS 记忆系统规范解析》第 1–2 章与第 4.1 节。

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.types import JSON

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# ORM table definitions
# ------------------------------------------------------------------

class EventRecord(Base):
    __tablename__ = "events"
    # 与 models.event.Event 字段一一对应，JSON 列存 summaries、role_list、source_events 等。

    event_id = Column(String, primary_key=True)
    subject_id = Column(String, default="")
    create_time = Column(DateTime, nullable=False)
    occurred_at = Column(DateTime, nullable=True)
    source_ids = Column(JSON, default=list)
    content_raw = Column(Text, nullable=False)
    summaries = Column(JSON, default=dict)
    summary_lengths = Column(JSON, default=dict)
    actual_max_level = Column(Integer, default=0)
    role_list = Column(JSON, default=list)
    is_abstract = Column(Boolean, default=False)
    is_abstracted = Column(Boolean, default=False)
    # 抽象覆盖累加量；合成更高阶抽象时对成员基本事件及中间抽象节点递增，检索时指数降权。
    abstract_coverage = Column(Float, default=0.0)
    status = Column(String, default="active")
    decoration = Column(Text, nullable=True)
    insight = Column(Text, nullable=True)
    event_length = Column(Integer, default=0)
    abstraction_level = Column(Integer, nullable=True)
    source_events = Column(JSON, nullable=True)
    is_tombstoned = Column(Boolean, default=False)
    activation_energy = Column(Float, default=0.0)  # 白皮书 2.5 记忆初始值硬绑定
    compression_ratio = Column(Float, default=0.0)  # 白皮书 1.2 封存后实际 sum_len/raw_len
    # 80/20 强制分裂链路（2026-05）：前后向指针，回忆时用来把前缀事件拉进回忆块。
    split_successor_event_ids = Column(JSON, default=list)
    split_prefix_event_ids = Column(JSON, default=list)
    ptsd_immune = Column(Boolean, default=False)
    origin = Column(String, default="external")
    location = Column(String, nullable=True)
    emotion = Column(JSON, nullable=True)
    forgetting_factor = Column(Float, default=1.0)
    seal_reason = Column(String, default="closed")
    recall_metadata = Column(JSON, default=dict)


class EventTier1Record(Base):
    """Tier-1 active pool metadata: A-Res sample_key + ASF (§4.5)."""

    __tablename__ = "event_tier1"

    event_id = Column(String, ForeignKey("events.event_id"), primary_key=True)
    sample_key = Column(Float, nullable=False, default=0.0, index=True)
    sample_u = Column(Float, nullable=False, default=0.5)
    asf_i = Column(Float, nullable=False, default=0.0)
    w_i_cached = Column(Float, nullable=False, default=0.0)
    w_eff_cached = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False)


class RoleRecord(Base):
    __tablename__ = "roles"

    role_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    entity_type = Column(String, default="person")
    aliases = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False)
    # 角色在系统中是否属于强制生成的“可疑记录”（身份不确切等）
    is_suspicious = Column(Boolean, default=False)


class WhitePaintingRecord(Base):
    __tablename__ = "white_painting_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String, default="")
    role_id = Column(String, ForeignKey("roles.role_id"), nullable=False, index=True)
    event_id = Column(String, ForeignKey("events.event_id"), nullable=False)
    role_summary = Column(Text, nullable=False)
    # 匠石视角分级白描（design/610）：L1 提及 / L2 互动 / L3 决策意图。
    l1_mention = Column(Text, nullable=True)
    l2_interaction = Column(Text, nullable=True)
    l3_decision = Column(Text, nullable=True)
    emotional_model = Column(JSON, default=dict)
    importance = Column(String, default="C")
    create_time = Column(DateTime, nullable=False)
    # 由 AE 映射的记忆权重 [0,1]，越高越抗遗忘
    memory_weight = Column(Float, default=0.0)
    # 动态遗忘因子，低于 0.02 则静默
    forgetting_factor = Column(Float, default=1.0)
    # 基础遗忘因子
    base_forgetting_factor = Column(Float, default=1.0)
    # 上次计算/访问时间
    last_accessed_time = Column(DateTime, nullable=False)
    # 该条目是否包含可疑标记
    is_suspicious = Column(Boolean, default=False)


class ShadowRecord(Base):
    __tablename__ = "shadow"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String, default="")
    content = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, nullable=False)
    # 按发生顺序的缓存句：[{"text": "...", "residual_id": "UC-..."}]
    buffer_items = Column(JSON, default=list)


class UnclosedEventRecord(Base):
    __tablename__ = "unclosed_events"

    id = Column(String, primary_key=True)
    subject_id = Column(String, default="")
    content_fragments = Column(JSON, default=list)
    interlocutor_attributions = Column(JSON, default=list)
    buffer_items = Column(JSON, default=list)
    identified_roles = Column(JSON, default=list)
    logical_gaps = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    last_hit_time = Column(DateTime, nullable=False)
    # 80/20 强制分裂：前缀事件链（自远而近）。UC 闭环为事件时由 MetabolismService 继承给目标事件。
    split_prefix_event_ids = Column(JSON, default=list)
    # 审计：评估器判定 oversized 但修复失败时置为 True；不触发强制封存。
    oversized = Column(Boolean, default=False)
    formation_role = Column(String, default="residual")


class ObjectMemoryEntryRecord(Base):
    """对象记忆时间线（替代 roles + white_painting_entries；无情感，见 design/410/810）。"""

    __tablename__ = "object_memory_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String, default="", index=True)
    object_id = Column(String, nullable=False, index=True)
    name = Column(String, nullable=True)  # 呈现用（与输入映射表一致，便于阅读，如回忆时）
    event_id = Column(String, ForeignKey("events.event_id"), nullable=False)
    summary = Column(Text, nullable=False)
    create_time = Column(DateTime, nullable=False)


class StoredMarkRecord(Base):
    """stored_marks 台账：segment_id（经历段 id）→ 本轮封存事件 id 列表（30 推进游标与台账用，见 design/210/810）。"""

    __tablename__ = "stored_marks"

    subject_id = Column(String, default="", index=True)
    input_id = Column(String, primary_key=True)  # 存经历段 segment_id（列名保留兼容）
    event_ids = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False)


class RecallTraceRecord(Base):
    """recall 轨迹：一次 recall 调用的输入 + 命中的回忆条目（observability / 召回质量回溯）。

    记录每次召回的输入（query / object_id / level / limit / anchor_event_ids）与命中条目
    （event_id / object_id / score / summary_level / source_ids / 摘要文本），
    供回溯"这条记忆属于谁"（诊断张冠李戴）。只作可观测，不影响召回结果。
    """

    __tablename__ = "recall_traces"

    recall_id = Column(String, primary_key=True)
    subject_id = Column(String, default="", index=True)
    query = Column(Text, nullable=False, default="")
    object_id = Column(String, nullable=True)
    level = Column(Integer, default=1)
    limit = Column(Integer, nullable=True)
    anchor_event_ids = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False)
    # 命中条目：RecallTraceItem.model_dump(mode="json") 列表。
    items = Column(JSON, default=list)
    n_items = Column(Integer, default=0)


class PortraitRecord(Base):
    """per-object 人物肖像：10 级渐进人物摘要（design/1010 §8.4 portrait）。"""

    __tablename__ = "object_portraits"

    subject_id = Column(String, default="", index=True)
    object_id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    # {"L1": {"level","text","budget","actual_len"}, ...}
    levels = Column(JSON, default=dict)
    max_level = Column(String, default="")
    total_content_len = Column(Integer, default=0)
    compression_ratio = Column(Float, default=0.0)
    fatigue = Column(Float, default=1.0)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class _ObjectKnowledgeRecord(Base):
    """匠石以后写回的对象认识。本仓只建表，不写入。"""

    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String, default="", index=True)
    object_id = Column(String, nullable=False, index=True)
    body = Column(Text, nullable=False, default="")
    source_event_ids = Column(JSON, default=list)
    is_tombstoned = Column(Boolean, default=False)
    forgetting_factor = Column(Float, default=1.0)
    created_at = Column(DateTime, nullable=False)


class ObjectTraitRecord(_ObjectKnowledgeRecord):
    """长期特征。空表，供匠石以后追加。"""

    __tablename__ = "object_traits"


class ObjectStateRecord(_ObjectKnowledgeRecord):
    """当前状态。空表，供匠石以后追加。"""

    __tablename__ = "object_states"


class ObjectDispositionRecord(_ObjectKnowledgeRecord):
    """互动倾向。空表，供匠石以后追加。"""

    __tablename__ = "object_dispositions"


class PortraitSummaryRecord(Base):
    """一条对象人物摘要（待并入或已并入肖像；全量重建用全部，增量用最长级+未并入）。"""

    __tablename__ = "portrait_summaries"

    summary_id = Column(String, primary_key=True)
    subject_id = Column(String, default="", index=True)
    object_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    source_event_id = Column(String, nullable=True)
    weight = Column(Float, default=0.5)
    incorporated = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)


# ------------------------------------------------------------------
# Database facade
# ------------------------------------------------------------------

import json
from functools import partial

# ------------------------------------------------------------------
# Database facade
# ------------------------------------------------------------------

class Database:
    def __init__(self, url: str):
        # 强制 json_serializer 使用 ensure_ascii=False，确保中文在 SQLite 数据库中以明文存储
        connect_args = {"timeout": 30} if url.startswith("sqlite") else {}
        self.engine = create_engine(
            url, 
            echo=False, 
            json_serializer=partial(json.dumps, ensure_ascii=False),
            connect_args=connect_args,
        )
        self._session_factory = sessionmaker(bind=self.engine)
        self._enable_sqlite_concurrency()

    def _enable_sqlite_concurrency(self) -> None:
        """SQLite 并发调优：WAL（写不阻塞读）+ busy_timeout + NORMAL 同步。

        ingest（后台线程）与 recall（主线程）并发时，避免 ``database is locked``；
        WAL 为持久设置，busy_timeout 通过 sqlite3 timeout=30 兜底。
        """
        if not self.engine.url.get_backend_name().startswith("sqlite"):
            return
        from sqlalchemy import text

        try:
            with self.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA busy_timeout=30000"))
                conn.execute(text("PRAGMA synchronous=NORMAL"))
        except Exception:  # noqa: BLE001
            logger.debug("SQLite WAL/busy_timeout setup skipped", exc_info=True)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)
        self._migrate_missing_columns()

    def _migrate_missing_columns(self) -> None:
        """Lightweight in-place migration for new columns introduced after initial schema.

        ``SQLAlchemy.create_all`` 只建新表，不会向已有表补列；dev 数据库在本地长期
        运行，新字段（如 80/20 分裂的链路字段）需要手动补列。此处只处理**追加列**
        这一种极窄的 migration 场景：
            - 逐字段 ``PRAGMA table_info`` 检查；
            - 缺失就 ``ALTER TABLE ... ADD COLUMN``；
            - 忽略除 SQLite 以外的后端（生产建议走正规 migration 工具）。
        """
        from sqlalchemy import inspect, text

        if not self.engine.url.get_backend_name().startswith("sqlite"):
            return

        expected: dict[str, list[tuple[str, str]]] = {
            "events": [
                ("split_successor_event_ids", "TEXT DEFAULT '[]'"),
                ("split_prefix_event_ids", "TEXT DEFAULT '[]'"),
                ("abstract_coverage", "FLOAT DEFAULT 0.0"),
                ("ptsd_immune", "BOOLEAN DEFAULT 0"),
                ("origin", "TEXT DEFAULT 'normal'"),
                ("recall_metadata", "TEXT DEFAULT '{}'"),
                ("subject_id", "TEXT DEFAULT ''"),
                ("occurred_at", "DATETIME"),
                ("source_ids", "TEXT DEFAULT '[]'"),
                ("emotion", "TEXT"),
                ("location", "TEXT"),
                ("forgetting_factor", "FLOAT DEFAULT 1.0"),
                ("seal_reason", "TEXT DEFAULT 'closed'"),
            ],
            "unclosed_events": [
                ("split_prefix_event_ids", "TEXT DEFAULT '[]'"),
                ("interlocutor_attributions", "TEXT DEFAULT '[]'"),
                ("buffer_items", "TEXT DEFAULT '[]'"),
                ("oversized", "BOOLEAN DEFAULT 0"),
                ("subject_id", "TEXT DEFAULT ''"),
                ("formation_role", "TEXT DEFAULT 'residual'"),
            ],
            "shadow": [
                ("subject_id", "TEXT DEFAULT ''"),
                ("buffer_items", "TEXT DEFAULT '[]'"),
            ],
            "roles": [
                ("subject_id", "TEXT DEFAULT ''"),
            ],
            "white_painting_entries": [
                ("subject_id", "TEXT DEFAULT ''"),
                ("l1_mention", "TEXT"),
                ("l2_interaction", "TEXT"),
                ("l3_decision", "TEXT"),
            ],
        }

        inspector = inspect(self.engine)
        with self.engine.begin() as conn:
            for table_name, cols in expected.items():
                if not inspector.has_table(table_name):
                    continue
                existing = {c["name"] for c in inspector.get_columns(table_name)}
                for name, ddl in cols:
                    if name in existing:
                        continue
                    conn.execute(text(
                        f'ALTER TABLE {table_name} ADD COLUMN {name} {ddl}'
                    ))

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._session_factory()
        try:
            yield s
        finally:
            s.close()
