#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆存储测试程序（匠石侧输入包装 + 落库 + 过程痕迹）。

用途
----
在独立目录 ``tests/memory_storage/`` 下运行一次记忆写入（ingest）：

- 输入按匠石侧协议包装：对话轮次（``匠石`` / ``deepseek``）→ ``MemoryExperience``；
- deepseek 只取每一轮的最终总结性回复（有长有短都按原文计入），
  判定看回复性质而非"文件里最后一条"；中间过程不计入；
- 累计对话正文达到阈值（默认 1000 字）才触发 ingest，否则提示等待下一轮；
- 正式数据（事件 / 白描 white_painting_entries / stored_marks / 影子 / 未完成事件 / 向量索引）与
  测试痕迹（LLM 请求、skill 调用）都落在同一 SQLite 库里，便于直接查看。

用法
----
::

    # 追加一轮对话（可多次），然后按阈值判断是否 ingest
    python tests/memory_storage/run_storage_test.py --add 匠石 "..." --add deepseek "..."

    # 长内容：从文件读取（UTF-8）
    python tests/memory_storage/run_storage_test.py --add-file 匠石 tmp/user.txt

    # 只查看当前累计状态（不建库、不 ingest）
    python tests/memory_storage/run_storage_test.py --list

    # 强制 ingest，离线演示用固定假模型（自检用）
    python tests/memory_storage/run_storage_test.py --force --fake

    # 真实 LLM：需要 API key（本 fork 无 .env，可 --env 指向原项目 .env）
    python tests/memory_storage/run_storage_test.py --force --env C:/path/to/rems_3/.env
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rems.config import REMSConfig  # noqa: E402
from rems.llm.provider import LLMProvider  # noqa: E402
from rems.pipeline import REMSPipeline  # noqa: E402
from rems.port import MemoryBatch, MemoryExperience  # noqa: E402
from rems.recall import (  # noqa: E402
    HashEmbedding,
    NullReranker,
    QdrantRecallVectorStore,
    RecallPipeline,
    SentenceTransformerEmbedding,
)
from rems.services.belief_revision_service import BeliefRevisionService  # noqa: E402
from rems.services.emotion_service import EMAEvolver  # noqa: E402
from rems.services.event_service import EventService  # noqa: E402
from rems.services.metabolism_service import MetabolismService  # noqa: E402
from rems.services.role_service import RoleService  # noqa: E402
from rems.skills.boundary_detection import BoundaryDetectionSkill  # noqa: E402
from rems.skills.event_enrichment import EventEnrichmentSkill  # noqa: E402
from rems.skills.role_extraction import RoleExtractionSkill  # noqa: E402
from rems.storage.database import Database  # noqa: E402
from rems.storage.repository import (  # noqa: E402
    EventRepository,
    MetabolismRepository,
    ObjectTimelineRepository,
    RoleRepository,
    StoredMarksRepository,
)
from rems.strategies.event_weight import EventWeightDeriver  # noqa: E402

SUBJECT_DEFAULT = "jshi-1"
THRESHOLD_DEFAULT = 1000
DB_NAME = "memory_storage_test.db"
QDRANT_DIR = "qdrant_test"
CONV_FILE = "conversation.jsonl"
REPORT_DIR = "run_reports"
SPEAKERS = {"匠石", "deepseek"}


# ---------------------------------------------------------------------------
# 对话输入（匠石侧包装）
# ---------------------------------------------------------------------------


def load_turns(conv_path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    if conv_path.exists():
        for line in conv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def append_turns(conv_path: Path, adds: list[tuple[str, str]]) -> list[dict[str, Any]]:
    turns = load_turns(conv_path)
    for speaker, content in adds:
        speaker = speaker.strip()
        if speaker not in SPEAKERS:
            raise SystemExit(f"未知说话人: {speaker}（应为 匠石 或 deepseek）")
        content = content.strip()
        if not content:
            raise SystemExit("内容不能为空")
        turns.append({
            "speaker": speaker,
            "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"),
        })
    conv_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in turns) + "\n",
        encoding="utf-8",
    )
    return turns


def pending_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """尚未摄入的轮次：conversation.jsonl 里没有 ``ingested_run`` 的轮次。"""
    return [t for t in turns if not t.get("ingested_run")]


def mark_ingested(conv_path: Path, turns: list[dict[str, Any]], run_id: str) -> None:
    """给本轮摄入的轮次标记 ``ingested_run=run_id``；只改匹配行，保留文件里其他轮次。"""
    matched = {(t.get("speaker"), t.get("content")) for t in turns}
    updated: list[dict[str, Any]] = []
    for t in load_turns(conv_path):
        if (
            not t.get("ingested_run")
            and (t.get("speaker"), t.get("content")) in matched
        ):
            t = {**t, "ingested_run": run_id}
        updated.append(t)
    conv_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in updated) + "\n",
        encoding="utf-8",
    )


def build_experiences(
    turns: list[dict[str, Any]], subject_id: str
) -> list[MemoryExperience]:
    """每轮对话 → 一段经历（design/210）：主体恒为匠石，deepseek 为对谈对象。"""
    exps: list[MemoryExperience] = []
    for i, t in enumerate(turns, 1):
        exps.append(MemoryExperience(
            subject_id=subject_id,
            text=f"{t['speaker']}: {t['content']}",
            objects={"deepseek": "OBJ-DEEPSEEK"},
            source_ids=(f"chat-20260823-{i:03d}",),
            segment_id=f"seg-{i:03d}",
            origin="external",
        ))
    return exps


def print_status(
    turns: list[dict[str, Any]],
    pending: list[dict[str, Any]],
    threshold: int,
) -> None:
    total_chars = sum(len(t["content"]) for t in turns)
    pending_chars = sum(len(t["content"]) for t in pending)
    print("=" * 72)
    print("对话输入状态（conversation.jsonl）")
    print(
        f"总轮次: {len(turns)}"
        f"（已摄入 {len(turns) - len(pending)} / 待摄入 {len(pending)}）"
    )
    print(
        f"累计正文 {total_chars} 字  |  待摄入 {pending_chars} 字"
        f"  |  阈值 {threshold} 字"
    )
    print("-" * 72)
    for i, t in enumerate(turns, 1):
        mark = "（已摄入）" if t.get("ingested_run") else ""
        preview = t["content"].replace("\n", " ")
        if len(preview) > 60:
            preview = preview[:60] + "…"
        print(f"  [{i:02d}] {t['speaker']:8s} ({len(t['content'])}字){mark} {preview}")
    if turns and turns[-1]["speaker"] != "deepseek":
        print("  （最后一轮 deepseek 总结回复尚未产生，未计入累计）")
    if pending_chars >= threshold:
        print("状态: 已达标，可以 ingest")
    else:
        print(f"状态: 待摄入未达标（还差 {threshold - pending_chars} 字），等待下一轮")


def load_env_file(path: Path) -> None:
    """加载 KEY=VALUE 格式的 env 文件（不覆盖已存在的环境变量）。"""
    if not path.exists():
        raise SystemExit(f"env 文件不存在: {path}")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 离线演示 LLM（--fake）
# ---------------------------------------------------------------------------


class CannedLLM:
    """固定假 LLM：boundary 封存整段输入；enrichment 返回固定摘要/情感/对象。"""

    def __init__(self, config: REMSConfig):
        self.config = config
        self._invocations: list[dict[str, Any]] = []
        self._last_input = ""
        self.calls: list[dict[str, Any]] = []

    def _get_model(self, task_type: str) -> str:
        mapping = self.config.llm.task_models
        return getattr(mapping, task_type, None) or mapping.default

    def _note(self, task_type: str, model: str) -> None:
        self._invocations.append({
            "task_type": task_type,
            "model": model,
            "latency_ms": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        })

    @staticmethod
    def _last_user(messages: list[dict[str, str]]) -> str:
        for m in reversed(messages or []):
            if m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    def _boundary_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        user = self._last_user(messages)
        lines = re.findall(r"^\s*\[(\d+)\]\s*(.*)$", user, re.MULTILINE)
        if not lines:
            return {"completed_events": [], "new_unclosed": []}
        indices = [int(i) for i, _ in lines]
        self._last_input = "".join(text for _, text in lines)
        return {
            "completed_events": [{"content_raw_indices": indices, "continuation_of": None}],
            "new_unclosed_indices": [],
        }

    def _enrichment_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        user = self._last_user(messages)
        m = re.search(r"## 事件原文\s*\n(.*?)\n\s*##", user, re.DOTALL)
        content = (m.group(1).strip() if m else "") or self._last_input or user[:120]
        preview = content[:60] + ("…" if len(content) > 60 else "")
        return {
            "summaries": {"L1": f"（离线演示摘要）{preview}"},
            "emotion": {"joy": 0.3, "trust": 0.2},
            "objects": [{"name": "deepseek", "summary": "离线演示：与匠石对话的 AI 对象。"}],
            "location": None,
            "keywords": ["测试", "记忆"],
        }

    def complete(self, task_type: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self._get_model(task_type)
        if task_type == "boundary_detection":
            data = self._boundary_payload(messages)
        elif task_type == "event_enrichment":
            data = self._enrichment_payload(messages)
        else:
            data = {}
        self._note(task_type, model)
        self.calls.append({"task_type": task_type, "model": model, "messages": list(messages)})
        return json.dumps(data, ensure_ascii=False)

    def complete_json(
        self, task_type: str, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        return json.loads(self.complete(task_type, messages, **kwargs))

    def complete_json_continue(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        new_user_content: str,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        messages.append({"role": "user", "content": new_user_content})
        raw = self.complete(task_type, messages, **kwargs)
        messages.append({"role": "assistant", "content": raw})
        return json.loads(raw), messages


# ---------------------------------------------------------------------------
# 测试痕迹（test_* 表，与正式 schema 分开）
# ---------------------------------------------------------------------------


class TraceDb:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS test_runs (
            run_id TEXT PRIMARY KEY,
            label TEXT,
            mode TEXT,
            started_at TEXT,
            finished_at TEXT,
            total_chars INTEGER,
            turn_count INTEGER,
            threshold INTEGER,
            subject_id TEXT,
            sealed_event_count INTEGER,
            error_count INTEGER,
            unclosed_count INTEGER,
            shadow_chars INTEGER,
            stored_mark_count INTEGER,
            qdrant_points INTEGER,
            elapsed_ms REAL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS test_experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            segment_id TEXT,
            speaker TEXT,
            chars INTEGER,
            text TEXT,
            objects_json TEXT,
            source_ids_json TEXT,
            origin TEXT,
            sealed_event_ids_json TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS test_llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            seq INTEGER,
            task_type TEXT,
            model TEXT,
            latency_ms REAL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_json TEXT,
            response TEXT
        );
        CREATE TABLE IF NOT EXISTS test_skill_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            seq INTEGER,
            skill TEXT,
            status TEXT,
            elapsed_ms REAL,
            detail_json TEXT
        );
        """)
        self.conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self.conn.execute(sql, params)
        self.conn.commit()

    def rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def run_start(
        self, run_id: str, *, label: str, mode: str, total_chars: int,
        turn_count: int, threshold: int, subject_id: str,
    ) -> None:
        self.execute(
            "INSERT INTO test_runs "
            "(run_id,label,mode,started_at,total_chars,turn_count,threshold,subject_id) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, label, mode, datetime.now().isoformat(timespec="seconds"),
             total_chars, turn_count, threshold, subject_id),
        )

    def run_end(self, run_id: str, **kw: Any) -> None:
        sets = ", ".join(f"{k}=?" for k in kw)
        self.execute(
            f"UPDATE test_runs SET {sets} WHERE run_id=?",
            (*kw.values(), run_id),
        )

    def add_experience(self, run_id: str, exp: MemoryExperience) -> None:
        self.execute(
            "INSERT INTO test_experiences "
            "(run_id,segment_id,speaker,chars,text,objects_json,source_ids_json,origin) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (run_id, exp.segment_id, exp.text.split(":", 1)[0], len(exp.text), exp.text,
             json.dumps(exp.objects, ensure_ascii=False),
             json.dumps(list(exp.source_ids), ensure_ascii=False), exp.origin),
        )

    def set_experience_result(
        self, run_id: str, segment_id: str, sealed_ids: list[str], error: str | None
    ) -> None:
        self.execute(
            "UPDATE test_experiences SET sealed_event_ids_json=?, error=? "
            "WHERE run_id=? AND segment_id=?",
            (json.dumps(sealed_ids, ensure_ascii=False), error, run_id, segment_id),
        )

    def add_llm(
        self, run_id: str, seq: int, task_type: str, model: str, latency_ms: float,
        tokens: dict[str, Any], messages: list[dict[str, str]], response: str,
    ) -> None:
        self.execute(
            "INSERT INTO test_llm_calls "
            "(run_id,seq,task_type,model,latency_ms,prompt_tokens,completion_tokens,"
            "total_tokens,prompt_json,response) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, seq, task_type, model, latency_ms,
             tokens.get("prompt_tokens"), tokens.get("completion_tokens"),
             tokens.get("total_tokens"),
             json.dumps(messages, ensure_ascii=False), response),
        )

    def add_skill(
        self, run_id: str, seq: int, skill: str, status: str,
        elapsed_ms: float, detail: dict[str, Any],
    ) -> None:
        self.execute(
            "INSERT INTO test_skill_steps "
            "(run_id,seq,skill,status,elapsed_ms,detail_json) VALUES (?,?,?,?,?,?)",
            (run_id, seq, skill, status, elapsed_ms,
             json.dumps(detail, ensure_ascii=False)),
        )

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# 插桩：记录 LLM 请求与 skill 调用
# ---------------------------------------------------------------------------


def wrap_llm(llm: Any, trace: TraceDb, run_id: str) -> Callable[[], None]:
    orig = llm.complete
    counter = {"n": 0}

    def wrapped(task_type: str, messages: list[dict[str, str]], **kwargs: Any) -> str:
        counter["n"] += 1
        seq = counter["n"]
        model = (
            llm._get_model(task_type)
            if hasattr(llm, "_get_model") else kwargs.get("model", "")
        )
        t0 = time.perf_counter()
        content = ""
        try:
            content = orig(task_type, messages, **kwargs)
        finally:
            elapsed = (time.perf_counter() - t0) * 1000.0
            inv = getattr(llm, "_invocations", None) or []
            m = inv[-1] if inv else None
            tokens = {
                k: (getattr(m, k, None) if m is not None else None)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            trace.add_llm(
                run_id, seq, task_type, model, round(elapsed, 2),
                tokens, messages, content,
            )
            print(
                f"  [LLM] #{seq:02d} {task_type} {model} "
                f"{elapsed:.0f}ms -> {len(content)} chars",
                flush=True,
            )
        return content

    setattr(llm, "complete", wrapped)
    return lambda: setattr(llm, "complete", orig)


def wrap_skill(
    obj: Any,
    method: str,
    skill_name: str,
    trace: TraceDb,
    run_id: str,
    summarize: Callable[[Any], dict[str, Any]],
) -> Callable[[], None]:
    orig = getattr(obj, method)
    counter = {"n": 0}

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        counter["n"] += 1
        seq = counter["n"]
        t0 = time.perf_counter()
        status, detail = "ok", {}
        try:
            result = orig(*args, **kwargs)
            detail = summarize(result)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            detail = {"error": str(exc)}
            raise
        finally:
            trace.add_skill(
                run_id, seq, skill_name, status,
                round((time.perf_counter() - t0) * 1000.0, 2), detail,
            )
        return result

    setattr(obj, method, wrapped)
    return lambda: setattr(obj, method, orig)


def _sum_boundary(result: Any) -> dict[str, Any]:
    return {
        "completed_count": len(result.completed_events),
        "new_unclosed_count": len(result.new_unclosed),
        "completed": [f.content_raw for f in result.completed_events],
        "new_unclosed": [u.content for u in result.new_unclosed],
    }


def _sum_enrichment(result: Any) -> dict[str, Any]:
    return {
        "summaries": result.summaries,
        "actual_max_level": result.actual_max_level,
        "emotion": result.emotion.model_dump(mode="json") if result.emotion else None,
        "object_snapshots": result.object_snapshots,
        "object_white_paintings": {
            k: v.model_dump(mode="json")
            for k, v in result.object_white_paintings.items()
        },
        "location": result.location,
        "keywords": result.keywords,
    }


def _sum_seal(ev: Any) -> dict[str, Any]:
    return {
        "event_id": ev.event_id,
        "content_raw": ev.content_raw,
        "summaries": ev.summaries,
        "roles": [
            {"role_id": r.role_id, "is_subject": r.is_subject}
            for r in ev.role_list
        ],
        "origin": ev.origin,
        "source_ids": ev.source_ids,
    }


# ---------------------------------------------------------------------------
# 管线装配（与 REMSPipeline.from_config 相同，仅注入自定义 llm）
# ---------------------------------------------------------------------------


def build_pipeline(config: REMSConfig, llm: Any) -> REMSPipeline:
    db = Database(config.storage.database_url)
    db.create_tables()

    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    meta_repo = MetabolismRepository(db)
    object_timeline_repo = ObjectTimelineRepository(db)
    stored_marks_repo = StoredMarksRepository(db)

    if config.embedding.provider == "hash":
        embedding = HashEmbedding()
    else:
        embedding = SentenceTransformerEmbedding(config.embedding.model_name)
    vector_store = QdrantRecallVectorStore(
        config.storage.qdrant_collection,
        url=config.storage.qdrant_url,
        path=config.storage.qdrant_path,
    )
    reranker: NullReranker = NullReranker()
    recall_pipeline = RecallPipeline(
        embedding, vector_store, event_repo, object_timeline_repo, reranker=reranker,
    )

    role_skill = RoleExtractionSkill(llm, config)
    boundary_skill = BoundaryDetectionSkill(llm, config)
    enrichment_skill = EventEnrichmentSkill(llm, config, role_fallback=role_skill)

    event_weight = EventWeightDeriver(config, event_repo, role_repo)
    emotion_evolver = EMAEvolver(config, role_repo)
    role_service = RoleService(config, role_repo, role_skill, llm=llm)
    event_service = EventService(
        config, llm, event_repo, None, enrichment_skill,
        role_service=role_service,
        emotion_evolver=emotion_evolver,
        event_weight=event_weight,
        object_timeline_repo=object_timeline_repo,
    )
    metabolism_service = MetabolismService.with_default_boundary_repair(
        config, meta_repo, boundary_skill, event_service,
        event_repo=event_repo, llm=llm,
    )
    belief_revision_service = BeliefRevisionService(config, event_repo, role_repo)

    return REMSPipeline(
        config=config, llm=llm, db=db,
        event_repo=event_repo, role_repo=role_repo, meta_repo=meta_repo,
        event_service=event_service, role_service=role_service,
        metabolism_service=metabolism_service,
        belief_revision_service=belief_revision_service,
        recall_pipeline=recall_pipeline,
        object_timeline_repo=object_timeline_repo,
        stored_marks_repo=stored_marks_repo,
    )


def build_config(args: argparse.Namespace, db_path: Path, qdrant_path: Path) -> REMSConfig:
    cfg = REMSConfig()
    cfg.storage.database_url = f"sqlite:///{db_path.as_posix()}"
    cfg.storage.qdrant_path = str(qdrant_path)
    cfg.storage.qdrant_url = ":memory:"
    cfg.embedding.provider = "hash" if args.embedding == "hash" else "local"
    if args.model:
        cfg.embedding.model_name = args.model
    cfg.subject_id = args.subject
    return cfg


# ---------------------------------------------------------------------------
# 落库审计与报告
# ---------------------------------------------------------------------------


def resolve_path(value: str | None, base: Path, default_name: str) -> Path:
    path = Path(value) if value else base / default_name
    return path if path.is_absolute() else base / path


def qdrant_points(pipeline: REMSPipeline) -> int:
    try:
        store = pipeline.recall_pipeline._vector_store
        res = store._client.count(collection_name=store._collection)
        return int(getattr(res, "count", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def table_counts(db_path: Path) -> dict[str, int]:
    tables = (
        "events", "roles", "white_painting_entries", "shadow", "unclosed_events",
        "object_memory_entries", "stored_marks",
        "test_runs", "test_experiences", "test_llm_calls", "test_skill_steps",
    )
    out: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for t in tables:
            try:
                out[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                out[t] = 0
    finally:
        conn.close()
    return out


def model_json(obj: Any) -> Any:
    try:
        return obj.model_dump(mode="json")
    except AttributeError:
        return obj


def build_report(
    run: dict[str, Any],
    experiences: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    skill_rows: list[dict[str, Any]],
    sealed_events: list[dict[str, Any]],
    wp_entries: list[dict[str, Any]],
    marks: dict[str, Any],
    shadow: Any,
    unclosed: list[dict[str, Any]],
    counts: dict[str, int],
    qdrant_n: int,
    errors: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# 记忆存储测试报告")
    lines.append("")
    lines.append(f"- run_id: {run['run_id']}")
    lines.append(f"- label: {run['label']}  |  mode: {run['mode']}")
    lines.append(f"- 开始: {run['started_at']}  |  结束: {run['finished_at']}")
    lines.append(
        f"- 输入: {run['total_chars']} 字 / {run['turn_count']} 轮 "
        f"（阈值 {run['threshold']}）"
    )
    lines.append(
        f"- 耗时: {run['elapsed_ms']:.0f} ms  |  封存事件: {run['sealed_event_count']}  "
        f"|  错误: {run['error_count']}"
    )
    lines.append("")

    lines.append("## 1. 输入包装（匠石侧）")
    lines.append("")
    lines.append("| segment | 说话人 | 字数 | sealed_event_ids | 错误 |")
    lines.append("|---|---|---|---|---|")
    for e in experiences:
        lines.append(
            f"| {e['segment_id']} | {e['speaker']} | {e['chars']} | "
            f"{e['sealed_event_ids_json'] or ''} | {e['error'] or ''} |"
        )
    lines.append("")

    lines.append("## 2. LLM 请求（skill 的模型调用）")
    lines.append("")
    for r in llm_rows:
        lines.append(f"### #{r['seq']:02d} {r['task_type']} ({r['model']})")
        lines.append("")
        lines.append(
            f"- 延迟: {r['latency_ms']:.0f} ms  |  tokens: "
            f"in={r['prompt_tokens'] or '?'} out={r['completion_tokens'] or '?'} "
            f"total={r['total_tokens'] or '?'}"
        )
        lines.append("")
        lines.append("**prompt（完整内容在 test_llm_calls.prompt_json）**")
        lines.append("")
        lines.append("```json")
        try:
            prompt = json.loads(r["prompt_json"])
            lines.append(json.dumps(prompt, ensure_ascii=False, indent=2)[:3000])
        except Exception:  # noqa: BLE001
            lines.append(str(r["prompt_json"])[:3000])
        lines.append("```")
        lines.append("")
        lines.append("**response**")
        lines.append("")
        lines.append("```")
        lines.append(r["response"])
        lines.append("```")
        lines.append("")

    lines.append("## 3. Skill 调用")
    lines.append("")
    for r in skill_rows:
        lines.append(
            f"- #{r['seq']:02d} **{r['skill']}** [{r['status']}] "
            f"{r['elapsed_ms']:.0f} ms"
        )
        try:
            detail = json.loads(r["detail_json"])
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(detail, ensure_ascii=False, indent=2)[:2000])
            lines.append("```")
        except Exception:  # noqa: BLE001
            pass
    lines.append("")

    lines.append("## 4. 落库结果")
    lines.append("")
    lines.append(f"### 封存事件（{len(sealed_events)} 条）")
    lines.append("")
    for ev in sealed_events:
        lines.append(f"- **{ev.get('event_id')}**  origin={ev.get('origin')}")
        lines.append(f"  - content_raw: {ev.get('content_raw')}")
        lines.append(f"  - summaries: {ev.get('summaries')}")
        lines.append(f"  - roles: {ev.get('role_list')}")
        lines.append(f"  - source_ids: {ev.get('source_ids')}  "
                     f"occurred_at: {ev.get('occurred_at')}")
    lines.append("")

    lines.append(f"### 白描 white_painting_entries（{len(wp_entries)} 条，匠石视角分级）")
    lines.append("")
    for o in wp_entries:
        lines.append(
            f"- 对象 {o.get('role_id')}  event={o.get('event_id')}  "
            f"subject={o.get('subject_id')}"
        )
        lines.append(f"  - L1 提及: {o.get('l1_mention')}")
        lines.append(f"  - L2 互动: {o.get('l2_interaction')}")
        lines.append(f"  - L3 决策: {o.get('l3_decision')}")
    lines.append("")

    lines.append("### stored_marks 台账")
    lines.append("")
    for seg, ids in marks.items():
        lines.append(f"- {seg} → {ids}")
    lines.append("")

    lines.append("### 影子 / 未完成事件")
    lines.append("")
    lines.append(f"- shadow_chars: {run['shadow_chars']}")
    lines.append(f"- unclosed_count: {run['unclosed_count']}")
    for u in unclosed:
        lines.append(f"  - {u.get('id')}: {u.get('content_fragments')}")
    lines.append("")

    lines.append("### 表行数")
    lines.append("")
    lines.append("| 表 | 行数 |")
    lines.append("|---|---|")
    for t, n in counts.items():
        lines.append(f"| {t} | {n} |")
    lines.append(f"| qdrant points（回忆索引） | {qdrant_n} |")
    lines.append("")

    if errors:
        lines.append("### 错误")
        lines.append("")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run_ingest(
    args: argparse.Namespace,
    base: Path,
    turns: list[dict[str, Any]],
    total_chars: int,
    conv_path: Path,
    all_turns: list[dict[str, Any]],
) -> int:
    run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
    db_path = resolve_path(args.db, base, DB_NAME)
    qdrant_path = resolve_path(args.qdrant_path, base, QDRANT_DIR)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    qdrant_path.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args, db_path, qdrant_path)
    mode = "fake" if args.fake else "real"
    if not args.fake and not cfg.llm.api_key:
        raise SystemExit(
            "未找到 LLM API key（真实模式需要）。\n"
            "  方式1: --env C:/路径/rems_3/.env （原项目有 REMS_LLM__API_KEY）\n"
            "  方式2: 把 .env 放到仓库根目录\n"
            "  方式3: 离线演示用 --fake"
        )
    llm: Any = CannedLLM(cfg) if args.fake else LLMProvider(cfg)

    trace = TraceDb(db_path)
    trace.run_start(
        run_id, label=args.label or "conversation-ingest", mode=mode,
        total_chars=total_chars, turn_count=len(turns),
        threshold=args.threshold, subject_id=args.subject,
    )

    experiences = build_experiences(turns, args.subject)
    for exp in experiences:
        trace.add_experience(run_id, exp)

    print("\n装配管线并 ingest ...")
    pipeline = build_pipeline(cfg, llm)
    restores = [
        wrap_llm(llm, trace, run_id),
        wrap_skill(
            pipeline.metabolism_service._boundary, "detect",
            "BoundaryDetectionSkill", trace, run_id, _sum_boundary,
        ),
        wrap_skill(
            pipeline.event_service._enrichment_skill, "enrich",
            "EventEnrichmentSkill", trace, run_id, _sum_enrichment,
        ),
        wrap_skill(
            pipeline.event_service, "seal_event",
            "EventService.seal_event", trace, run_id, _sum_seal,
        ),
    ]
    try:
        t0 = time.perf_counter()
        batch = MemoryBatch(subject_id=args.subject, experiences=tuple(experiences))
        result = pipeline.ingest_batch(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        for restore in restores:
            restore()

    for exp in experiences:
        sealed_ids = result.stored_marks.get(exp.segment_id, [])
        error = next(
            (e for e in result.errors if e.startswith(f"segment {exp.segment_id}")),
            None,
        )
        trace.set_experience_result(run_id, exp.segment_id, sealed_ids, error)

    # 落库审计
    sealed_events: list[dict[str, Any]] = []
    wp_entries: list[dict[str, Any]] = []
    for sid in result.sealed_event_ids:
        ev = pipeline.event_repo.get(sid)
        if ev is None:
            continue
        sealed_events.append(model_json(ev))
        for re_ in ev.role_list:
            if not re_.is_subject and re_.role_id:
                wp = pipeline.role_repo.get_white_painting_by_event(re_.role_id, sid)
                if wp:
                    wp_entries.append(model_json(wp))
    marks = {
        exp.segment_id: pipeline.stored_marks_repo.get(exp.segment_id)
        for exp in experiences
    }
    shadow = pipeline.meta_repo.get_shadow()
    unclosed = [model_json(u) for u in pipeline.meta_repo.get_unclosed_events()]
    qdrant_n = qdrant_points(pipeline)
    counts = table_counts(db_path)

    trace.run_end(
        run_id,
        finished_at=datetime.now().isoformat(timespec="seconds"),
        sealed_event_count=len(result.sealed_event_ids),
        error_count=len(result.errors),
        unclosed_count=len(unclosed),
        shadow_chars=shadow.length,
        stored_mark_count=len(result.stored_marks),
        qdrant_points=qdrant_n,
        elapsed_ms=round(elapsed_ms, 2),
        notes="; ".join(result.errors) if result.errors else None,
    )

    run_row = trace.rows("SELECT * FROM test_runs WHERE run_id=?", (run_id,))[0]
    exp_rows = trace.rows(
        "SELECT * FROM test_experiences WHERE run_id=? ORDER BY id", (run_id,)
    )
    llm_rows = trace.rows(
        "SELECT * FROM test_llm_calls WHERE run_id=? ORDER BY seq", (run_id,)
    )
    skill_rows = trace.rows(
        "SELECT * FROM test_skill_steps WHERE run_id=? ORDER BY id", (run_id,)
    )
    trace.close()

    report = build_report(
        run_row, exp_rows, llm_rows, skill_rows, sealed_events,
        wp_entries, marks, shadow, unclosed, counts, qdrant_n, result.errors,
    )
    report_dir = base / REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_id}.md"
    report_path.write_text(report, encoding="utf-8")

    if args.db is None:
        # 仅默认主测试库摄入时标记，自检（--db）不污染对话状态
        mark_ingested(conv_path, all_turns, run_id)
        print(f"已标记本轮 {len(turns)} 轮为已摄入（ingested_run={run_id}）")

    # 控制台摘要
    print()
    print("=" * 72)
    print("运行摘要")
    print(f"run_id        : {run_id}  ({mode})")
    print(f"输入          : {total_chars} 字 / {len(turns)} 轮")
    print(f"封存事件      : {len(result.sealed_event_ids)}")
    print(f"stored_marks  : {result.stored_marks}")
    print(f"白描条目      : {len(wp_entries)} 条")
    print(f"未完成事件    : {len(unclosed)}  |  影子 {shadow.length} 字")
    print(f"向量索引      : {qdrant_n} 点")
    print(f"错误          : {result.errors or '无'}")
    print(f"耗时          : {elapsed_ms:.0f} ms")
    print("-" * 72)
    print(f"SQLite 数据库 : {db_path}")
    print(f"Qdrant 目录   : {qdrant_path}")
    print(f"报告          : {report_path}")
    print("=" * 72)
    print("查看建议:")
    print("  1. 用 SQLite 工具打开上面的 .db，events / object_memory_entries /")
    print("     stored_marks / shadow / unclosed_events 是正式落库；")
    print("     test_llm_calls / test_skill_steps 是中间过程痕迹。")
    print("  2. 报告里已包含每轮 LLM 的 prompt/response 与 skill 摘要。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="记忆存储测试：匠石侧对话输入 → ingest → 落库 + 过程痕迹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--add", nargs=2, metavar=("SPEAKER", "CONTENT"),
        action="append", default=[], help="追加一轮对话（可多次）",
    )
    parser.add_argument(
        "--add-file", nargs=2, metavar=("SPEAKER", "PATH"),
        action="append", default=[], help="从 UTF-8 文件追加长内容",
    )
    parser.add_argument("--list", action="store_true", help="只查看累计状态")
    parser.add_argument("--force", action="store_true", help="忽略阈值，强制 ingest")
    parser.add_argument("--fake", action="store_true", help="离线固定假 LLM")
    parser.add_argument("--embedding", choices=["hash", "local"], default="hash")
    parser.add_argument("--model", default=None, help="嵌入模型名（local 时）")
    parser.add_argument("--threshold", type=int, default=THRESHOLD_DEFAULT)
    parser.add_argument("--subject", default=SUBJECT_DEFAULT, help="记忆主体（匠石）")
    parser.add_argument("--db", default=None, help="SQLite 路径（默认本目录）")
    parser.add_argument("--qdrant-path", default=None, help="Qdrant 本地目录")
    parser.add_argument("--env", default=None, help="加载 API key 等环境变量文件")
    parser.add_argument("--label", default=None, help="本次运行标签")
    parser.add_argument(
        "--mark-ingested", default=None, metavar="RUN_ID",
        help="把未摄入轮次标记为已摄入（回填用）",
    )
    parser.add_argument(
        "--reset-ingested", action="store_true",
        help="清除全部 ingested_run 标记（重新累计，配合重新跑批次）",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="只取前 N 轮（验证用，不影响文件里的其他轮次）",
    )
    parser.add_argument(
        "--last", type=int, default=None, metavar="N",
        help="只取最近 N 轮（按追加顺序取末尾，数据库累积不清库）",
    )
    args = parser.parse_args(argv)

    base = Path(__file__).resolve().parent
    conv_path = base / CONV_FILE
    turns = load_turns(conv_path)

    if args.add:
        turns = append_turns(conv_path, [tuple(a) for a in args.add])
    if args.add_file:
        adds: list[tuple[str, str]] = []
        for speaker, path in args.add_file:
            content = Path(path).read_text(encoding="utf-8").strip()
            adds.append((speaker, content))
        turns = append_turns(conv_path, adds)

    if args.limit:
        turns = turns[: args.limit]
    if args.last:
        turns = turns[-args.last:]

    if not turns:
        print(f"{conv_path} 为空：先用 --add / --add-file 追加对话轮次。")
        return 1

    if args.mark_ingested:
        mark_ingested(conv_path, turns, args.mark_ingested)
        print(f"已将未摄入轮次标记为 ingested_run={args.mark_ingested}")
        return 0

    if args.reset_ingested:
        updated = [
            {k: v for k, v in t.items() if k != "ingested_run"}
            for t in turns
        ]
        conv_path.write_text(
            "\n".join(json.dumps(t, ensure_ascii=False) for t in updated) + "\n",
            encoding="utf-8",
        )
        print("已清除全部 ingested_run 标记，全部轮次重新计入待摄入。")
        return 0

    pending = pending_turns(turns)
    pending_chars = sum(len(t["content"]) for t in pending)
    print_status(turns, pending, args.threshold)
    if args.list:
        return 0
    if pending_chars < args.threshold and not args.force:
        print()
        print(f"待摄入 {pending_chars} 字 < 阈值 {args.threshold} 字，等待下一轮对话。")
        print(f"还差 {args.threshold - pending_chars} 字；下一轮消息到达后继续追加即可"
              "（deepseek 只取最终的总结性回复）。")
        return 0

    if args.env:
        load_env_file(Path(args.env))
    return run_ingest(args, base, pending, pending_chars, conv_path, turns)


if __name__ == "__main__":
    raise SystemExit(main())
