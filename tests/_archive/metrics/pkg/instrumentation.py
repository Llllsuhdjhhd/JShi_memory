"""Test-only monkey patches: LLM calls, vector search, hybrid recall scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from rems.models.event import Event
from rems.services.recall_service import RecallService


@dataclass
class LLMCallRecord:
    task_type: str
    system_head: str
    prompt_chars: int
    reply_preview: str
    reply_chars: int
    seconds: float


@dataclass
class VectorSearchRecord:
    query_head: str
    n_results: int
    hit_count: int
    seconds: float


@dataclass
class HybridScoreRecord:
    event_id: str
    cosine_sim: float
    cosine_w: float
    contrib_cosine: float
    time_decay: float
    contrib_time: float
    role_boost: float
    contrib_role: float
    ae: float
    contrib_ae: float
    act: float
    contrib_act: float
    total: float


@dataclass
class InstrumentationStore:
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    vector_searches: list[VectorSearchRecord] = field(default_factory=list)
    hybrid_scores: list[HybridScoreRecord] = field(default_factory=list)
    # First role_extraction system prompt (full head) for §2.2
    first_role_extraction_system: str | None = None
    # Abstraction task user messages (includes anchor labels; one per call)
    abstraction_user_messages: list[str] = field(default_factory=list)


def install_llm_tracer(store: InstrumentationStore, llm: Any) -> Callable[[], None]:
    """Patch ``llm.complete`` to record calls. Returns uninstall function."""
    _orig = llm.complete

    def _complete(
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> str:
        sys_head = ""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content") or ""
                sys_head = content[:500]
                if task_type == "role_extraction" and store.first_role_extraction_system is None:
                    store.first_role_extraction_system = content[:800]
                break
        pch = sum(len(m.get("content") or "") for m in messages)
        user_blob = ""
        for m in messages:
            if m.get("role") == "user":
                user_blob = m.get("content") or ""
                break
        if task_type == "abstraction":
            store.abstraction_user_messages.append(user_blob[:8000])

        t0 = time.perf_counter()
        try:
            out = _orig(task_type, messages, temperature=temperature)
        finally:
            dt = time.perf_counter() - t0

        tail = (out or "").replace("\r", " ").replace("\n", " ").strip()
        if len(tail) > 72:
            tail = tail[:71] + "…"
        store.llm_calls.append(
            LLMCallRecord(
                task_type=task_type,
                system_head=sys_head,
                prompt_chars=pch,
                reply_preview=tail,
                reply_chars=len(out or ""),
                seconds=dt,
            ),
        )
        return out

    llm.complete = _complete  # type: ignore[method-assign]

    def _uninstall() -> None:
        llm.complete = _orig  # type: ignore[method-assign]

    return _uninstall


def install_vector_tracer(store: InstrumentationStore, vector: Any) -> Callable[[], None]:
    """Patch ``vector_store.search``."""
    _orig = vector.search

    def _search(query: str, n_results: int = 10, where: dict[str, Any] | None = None):
        t0 = time.perf_counter()
        hits = _orig(query, n_results=n_results, where=where)
        dt = time.perf_counter() - t0
        qh = (query or "")[:240].replace("\n", " ")
        store.vector_searches.append(
            VectorSearchRecord(
                query_head=qh,
                n_results=n_results,
                hit_count=len(hits or []),
                seconds=dt,
            ),
        )
        return hits

    vector.search = _search  # type: ignore[method-assign]

    def _uninstall() -> None:
        vector.search = _orig  # type: ignore[method-assign]

    return _uninstall


def install_recall_scoring_tracer(
    store: InstrumentationStore,
    recall: RecallService,
) -> Callable[[], None]:
    """Patch ``RecallService._hybrid_score`` to log weighted components."""

    _orig = recall._hybrid_score

    def _wrapped(event: Event, cosine_sim: float) -> float:
        cfg = recall._config
        time_decay = RecallService._time_decay(event.create_time)

        role_boost = 0.0
        importance_weights = {"S": 0.3, "A": 0.2, "B": 0.1, "C": 0.05, "D": 0.0}
        for re in event.role_list:
            key = re.importance.value if hasattr(re.importance, "value") else str(re.importance)
            role_boost = max(role_boost, importance_weights.get(key, 0.0))

        ae = event.affective_energy
        act = event.activation_energy
        ae_w = cfg.ae_score_weight
        act_w = cfg.activation_energy_weight
        cosine_w = max(0.0, 1.0 - ae_w - act_w - 0.15 - 0.20)

        c_cos = cosine_w * cosine_sim
        c_time = 0.15 * time_decay
        c_role = 0.20 * role_boost
        c_ae = ae_w * ae
        c_act = act_w * act
        total = c_cos + c_time + c_role + c_ae + c_act

        store.hybrid_scores.append(
            HybridScoreRecord(
                event_id=event.event_id,
                cosine_sim=cosine_sim,
                cosine_w=cosine_w,
                contrib_cosine=c_cos,
                time_decay=time_decay,
                contrib_time=c_time,
                role_boost=role_boost,
                contrib_role=c_role,
                ae=ae,
                contrib_ae=c_ae,
                act=act,
                contrib_act=c_act,
                total=total,
            ),
        )
        return _orig(event, cosine_sim)

    recall._hybrid_score = _wrapped  # type: ignore[method-assign]

    def _uninstall() -> None:
        recall._hybrid_score = _orig  # type: ignore[method-assign]

    return _uninstall


def format_hybrid_line(rec: HybridScoreRecord) -> str:
    return (
        f"`{rec.event_id}` "
        f"cosine:{rec.contrib_cosine:.3f} | time:{rec.contrib_time:.3f} | "
        f"role:{rec.contrib_role:.3f} | ae:{rec.contrib_ae:.3f} | act:{rec.contrib_act:.3f} "
        f"→ total={rec.total:.3f}"
    )
