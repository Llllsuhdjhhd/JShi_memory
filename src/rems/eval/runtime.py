"""Real JShi text-entry core, ledger, memory control and REMS adapter.

History is teacher-forced (no synthetic replies); each question gets a fresh JShi
process and a copy of the immutable, closed REMS baseline. Voice/JEV routing and
background long-term-experience maintenance are explicitly outside this baseline.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime as RealDateTime
from pathlib import Path

from .dataset import sessions, turn_text

SUBJECT = "jshi-eval-observer"
ASKER = "eval-questioner"


@contextmanager
def evaluation_clock(date):
    """Patch application clock references, never stdlib/Pydantic/ML or network clocks."""
    class DomainDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return date.replace(tzinfo=tz) if tz is not None else date.replace(tzinfo=None)
        @classmethod
        def utcnow(cls):
            return date.replace(tzinfo=None)
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            from pydantic_core import core_schema
            return core_schema.datetime_schema()
    changed = []
    for name, module in list(sys.modules.items()):
        if name.startswith(("rems.", "jshi.")) and module is not None:
            if vars(module).get("datetime") is RealDateTime:
                changed.append(module)
                module.datetime = DomainDateTime
    try:
        yield
    finally:
        for module in changed:
            module.datetime = RealDateTime


def load_jshi(path):
    src = Path(path).resolve() / "src"
    if not (src / "jshi" / "subject" / "process.py").is_file():
        raise ValueError("JShi source checkout is required; no toy-core fallback")
    sys.path.insert(0, str(src))


def delivered_memory(user_text):
    """Extract the rendered sections actually sent to the core, not backend raw text."""
    headings = {"回忆", "你此时的回忆", "素材·回忆", "人物肖像·可修订", "人物相处经验·可修订",
                "素材·人物肖像（可修订）", "素材·人物相处经验（可修订）"}
    matches = list(re.finditer(r"(?m)^【([^\n】]+)】[^\n]*\n", user_text))
    sections = []
    for i, match in enumerate(matches):
        if match[1] in headings:
            end = matches[i+1].start() if i+1 < len(matches) else len(user_text)
            sections.append(user_text[match.start():end].strip())
    return "\n\n".join(sections)


class CoreTransport:
    def __init__(self, llm):
        self.llm, self.name, self.captures = llm, "deepseek-flash", []

    def generate(self, request):
        from jshi.models import ModelResponse
        from jshi.models.prompt import build_system, build_user
        user = build_user(request)
        messages = [{"role": "system", "content": build_system(request) + "\n"
            "For this English dialogue task, give a concise English reply. The two named "
            "speakers are people whose conversations you observed, not yourself. "
            "When history is insufficient, say 'Not mentioned' rather than inventing facts. "
            "Keep the required JSON response schema."}, {"role": "user", "content": user}]
        capture = {"purpose": request.purpose, "user": user,
                   "as_of": request.now.isoformat() if request.now else None,
                   "delivered_memory": delivered_memory(user), "context": list(request.context)}
        self.captures.append(capture)
        raw = self.llm.complete("core", messages, json_output=bool(
            getattr(request, "json_output", False) or getattr(request, "persona_schema", None)))
        return ModelResponse(text=raw, model=self.name)


def make_process(root, pipe, llm, names, *, recall=True):
    from jshi.identity import IdentityProfile, IdentityRepository
    from jshi.subject import SubjectProcess, SubjectRepository
    from jshi.skill import CognitionSkill, SkillModelPort
    from jshi.memory import MemoryShell
    from jshi.memory.rems3 import Rems3MemoryBackend
    from jshi.recognition import ObjectProfile
    from jshi.style import StylePackStore, ZoneStore
    from jshi.memory.traces import JsonlRecallTraceStore
    from rems.port import MemoryBatch, MemoryExperience

    root.mkdir(parents=True, exist_ok=True)
    identities = IdentityRepository(root / "identities.json")
    identities.create(IdentityProfile(SUBJECT, "匠石", "LoCoMo observer evaluation",
        "I am JShi. I observed conversations between other people; their experiences are not my biography."))
    backend = Rems3MemoryBackend(pipe, data_dir=root / "adapter",
                                 engine_types=(MemoryBatch, MemoryExperience))
    if backend._semantic_disabled and pipe.config.embedding.provider != "hash":
        raise RuntimeError("JShi disabled semantic retrieval; encoder must be installed before evaluation")
    # First-round event-memory baseline: prevent opportunistic unjoined background jobs.
    backend.long_term_experience = None
    backend._portrait_refresh = None
    shell = MemoryShell(backend)
    transport = CoreTransport(llm)
    process = SubjectProcess(SubjectRepository(root / "subject.sqlite3"), identities,
        SkillModelPort(CognitionSkill(transport), streaming=False), memory=shell,
        style_packs=StylePackStore(root / "styles.json"), zone_store=ZoneStore(root / "zones.json"),
        recall_traces=JsonlRecallTraceStore(root / "recall_traces.jsonl"))
    # Default wood is the current text-entry core; no fabricated JEV decisions.
    for name, oid in {**names, "Evaluator": ASKER}.items():
        process.profiles.create(ObjectProfile(object_id=oid, label=name,
                                              source="benchmark", status="confirmed"))
    original = process.recognition
    class TextRecognition:
        name = "benchmark-explicit-name-entry"
        def resolve(self, subject_id, text, object_ref, channel=None, carriers=()):
            mentioned = tuple(oid for name, oid in names.items()
                if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I))
            return original.resolve(subject_id, text, object_ref, channel, carriers,
                                    mentioned_object_ids=mentioned)
    process.recognition = TextRecognition()
    if not recall:
        # Disable historical sources at assembly, retaining the identical core/question.
        process.assembler.sources = tuple(s for s in process.assembler.sources
            if s.name not in {"memory", "person_portrait", "person_experience"})
    return process, transport


def replay_history(pipe, llm, root, sample, *, chunk_chars=3000, default_triggers=False):
    from jshi.memorycontrol import InProcessMemoryControl
    names = {sample["conversation"][k]: f"{sample['sample_id']}:{k}"
             for k in ("speaker_a", "speaker_b")}
    process, _ = make_process(root, pipe, llm, names)
    class CheckedIngest:
        def ingest_batch(self, batch):
            result = process.memory.ingest_batch(batch)
            if result.errors:
                raise RuntimeError("Memory backend did not accept a clean batch: " + "; ".join(result.errors))
            return result
    trigger_args = {} if default_triggers else {"flush_max_chars": 1, "flush_max_segments": 1}
    # Dataset session clocks are naive local times, as are the supplied ledger
    # timestamps; keep the trigger clock in the same domain.
    control = InProcessMemoryControl(process.activity_ledger, CheckedIngest(), **trigger_args,
                                     now=lambda: date,
                                     subject_name=lambda _: "匠石")
    log = []
    known_events = {e.event_id for e in pipe.event_repo.list_all()}
    started = time.perf_counter()
    for index, date, turns in sessions(sample):
        with evaluation_clock(date):
            size = 0
            for n, turn in enumerate(turns):
                process.activity_ledger.append_external(SUBJECT,
                    actor_object_id=names[turn["speaker"]], text_raw=turn_text(turn),
                    objects=names, mentioned_object_ids=tuple(names.values()),
                    source_ids=(f"{sample['sample_id']}:{turn['dia_id']}",), occurred_at=date)
                size += len(turn_text(turn))
                if ((default_triggers and control.evaluate(SUBJECT).should_trigger)
                        or (not default_triggers and (size >= chunk_chars or n == len(turns)-1))):
                    result = control.run_once(SUBJECT)
                    # Production ingestion can catch backend exceptions internally.
                    # A transport failure must invalidate this replay immediately.
                    ledger = getattr(llm, "ledger", None)
                    if ledger is not None and ledger.summary()["unsettled"]:
                        raise RuntimeError("Interrupted paid write; discard incomplete baseline, retain budget reservation")
                    log.append({"session": index, "status": result.status, "chars": size,
                                "trigger": result.decision.reason})
                    if result.status != "ingested":
                        raise RuntimeError(f"History ingest failed: {result.error or result.status}")
                    # Pydantic's pre-bound datetime default factories deliberately
                    # remain untouched; stamp newly accepted events on the replay clock.
                    for event in pipe.event_repo.list_all():
                        if event.event_id not in known_events:
                            event.create_time = date
                            pipe.event_repo.save(event)
                            known_events.add(event.event_id)
                    size = 0
        print(f"ingest {sample['sample_id']} session {index}: {len(pipe.event_repo.list_all())} events", flush=True)
    # A finite dataset ends even when default delivery thresholds are not met.
    if default_triggers and control._pending(SUBJECT):
        control._flush_max_chars = 1
        result = control.run_once(SUBJECT)
        if result.status != "ingested":
            raise RuntimeError(f"Final pending delivery failed: {result.error}")
        log.append({"session": "end", "status": result.status, "trigger": "dataset_end_delivery"})
    info = {"log": log, "events": len(pipe.event_repo.list_all()),
            "default_jshi_trigger_conditions": default_triggers,
            "seconds": time.perf_counter()-started, "names": names,
            "last_date": sessions(sample)[-1][1].isoformat(),
            "sample_id": sample["sample_id"], "history_sessions": len(sessions(sample)),
            "history_turns": sum(len(s[2]) for s in sessions(sample))}
    (root / "history_complete.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    return finalize_history(pipe, info)


def finalize_history(pipe, info):
    """Close residual buffer after a persisted history-complete checkpoint."""
    from rems.port import MemoryExperience
    names = info["names"]
    last_date = RealDateTime.fromisoformat(info["last_date"])
    with evaluation_clock(last_date):
        memory = MemoryExperience(subject_id=SUBJECT, text="", objects=names,
                                  occurred_at=last_date, source_ids=())
        events = pipe.metabolism_service.process_input("", force_save=True, memory=memory)
        for event in events:
            event.create_time = last_date
            pipe.event_repo.save(event)
            pipe.recall_pipeline.index_event(event)
    return {**info, "events": len(pipe.event_repo.list_all()), "residual_events_closed": len(events)}


def answer_question(pipe, llm, root, names, question, date, seed, *, recall=True):
    started = time.perf_counter()
    random.seed(seed)
    # Construct first so first-use lazy application imports exist before clock
    # references are patched. No question or historical scene is processed here.
    process, transport = make_process(root, pipe, llm, names, recall=recall)
    previous_traces = {row["recall_id"] for row in pipe.recall_trace_repo.list_all(subject_id=SUBJECT)}
    with evaluation_clock(date):
        from jshi.memorycontrol import InProcessMemoryControl
        process.memory_control = InProcessMemoryControl(
            process.activity_ledger, process.memory, flush_max_chars=10**12,
            flush_max_segments=10**12, flush_max_idle_seconds=float("inf"),
            flush_on_idle_seconds=float("inf"), flush_night_window="",
        )
        candidates: list = []
        pipe.recall_candidate_sink = candidates
        try:
            result = process.experience(SUBJECT, question, object_ref=ASKER)
        finally:
            pipe.recall_candidate_sink = None
        if not transport.captures:
            raise RuntimeError("Real JShi cognition was not called")
        if len(transport.captures) != 1:
            raise RuntimeError("Unexpected extra cognition calls; inspect captured requests")
        capture = transport.captures[0]
        errors = [r.error for r in result.current_state.source_report if r.error]
        if errors:
            raise RuntimeError("Assembly source failed: " + "; ".join(errors))
        fallback = (process.last_model_response.metadata or {}).get("skill_fallback")
        if fallback:
            raise RuntimeError(f"Core schema fallback: {fallback}")
        record = {"answer": result.action_text, "delivered_memory": capture["delivered_memory"],
                  "seconds": time.perf_counter()-started,
                  "as_of": capture["as_of"],
                  "core_user": capture["user"], "core_context": capture["context"],
                  "sources": [{"source": r.source, "loaded": list(r.loaded_ids),
                               "skipped": list(r.skipped_ids)} for r in result.current_state.source_report],
                  # 后端候选的去向（排名/长度/淘汰原因），与 delivered_memory 对照可分离
                  # “没召回”“召回了没交付”“交付了没答对”。仅诊断，不参与评分。
                  "recall_candidates": [c.as_dict() for c in candidates]}
        # Evaluation-only provenance. Never pass raw events or gold evidence to
        # cognition. Ignore historical traces copied with the sealed baseline.
        retrieved = {}
        for trace in pipe.recall_trace_repo.list_all(subject_id=SUBJECT):
            if trace["recall_id"] in previous_traces:
                continue
            for item in trace["items"]:
                event = pipe.event_repo.get(item["event_id"])
                if event is not None:
                    retrieved[event.event_id] = {
                        "event_id": event.event_id, "source_ids": list(event.source_ids),
                        "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
                        "content_raw": event.content_raw,
                    }
        record["retrieved_events"] = list(retrieved.values())
        process.repository.close() if hasattr(process.repository, "close") else None
        return record


def close_pipeline(pipe):
    pipe.recall_pipeline._vector_store._client.close()
    pipe.db.engine.dispose()
