from __future__ import annotations

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


class UnclosedEventRecord(Base):
    __tablename__ = "unclosed_events"

    id = Column(String, primary_key=True)
    subject_id = Column(String, default="")
    content_fragments = Column(JSON, default=list)
    identified_roles = Column(JSON, default=list)
    logical_gaps = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    last_hit_time = Column(DateTime, nullable=False)
    # 80/20 强制分裂：前缀事件链（自远而近）。UC 闭环为事件时由 MetabolismService 继承给目标事件。
    split_prefix_event_ids = Column(JSON, default=list)
    # 审计：评估器判定 oversized 但修复失败时置为 True；不触发强制封存。
    oversized = Column(Boolean, default=False)


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
        self.engine = create_engine(
            url, 
            echo=False, 
            json_serializer=partial(json.dumps, ensure_ascii=False)
        )
        self._session_factory = sessionmaker(bind=self.engine)

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
            ],
            "unclosed_events": [
                ("split_prefix_event_ids", "TEXT DEFAULT '[]'"),
                ("oversized", "BOOLEAN DEFAULT 0"),
                ("subject_id", "TEXT DEFAULT ''"),
            ],
            "shadow": [
                ("subject_id", "TEXT DEFAULT ''"),
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
