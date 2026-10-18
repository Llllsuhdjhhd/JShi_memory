"""Age-based, chronological refinement. Source records are never overwritten."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

from ..models.event import Event
from ..skills.boundary_detection import BoundaryDetectionSkill
from ..utils.text import normalize_indices


def utc(value: datetime) -> datetime:
    # Explicit and host-independent for legacy timestamps without offsets.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def time_span(event: Event) -> tuple[datetime, datetime]:
    return (event.event_time_start or event.occurred_at or event.create_time,
            event.event_time_end or event.occurred_at or event.create_time)


class RefinementService:
    def __init__(self, config, event_repo, event_service, recall_pipeline, llm):
        self.config, self.repo, self.events = config, event_repo, event_service
        self.recall, self.llm = recall_pipeline, llm

    @staticmethod
    def source_text(event: Event) -> str:
        if event.memory_stage == 0:
            return event.content_raw
        text = event.summaries.get("L1", "")
        if not text.strip():
            raise ValueError(f"Missing prior L1: {event.event_id}; source remains visible")
        return text

    def select(self, subject_id: str, target: int, now: datetime) -> list[Event]:
        policy = self.config.memory_refinement
        cutoff = utc(now) - timedelta(days=policy.stages[target - 1].age_days)
        eligible = [event for event in self.repo.list_all()
                    if event.subject_id == subject_id and event.memory_stage == target - 1
                    and not event.refinement_hidden_by and not event.is_abstracted
                    and utc(time_span(event)[1]) <= cutoff]
        eligible.sort(key=lambda event: (utc(time_span(event)[0]), utc(time_span(event)[1]),
                                         utc(event.create_time), event.event_id))
        candidates, size = [], 0
        for event in eligible:
            length = len(self.source_text(event)) + (2 if candidates else 0)
            if candidates and size + length > policy.candidate_chars:
                break
            candidates.append(event)
            size += length
        ordered = list(reversed(candidates)) if policy.direction == "backward" else candidates
        selected, size = [], 0
        for event in ordered:
            length = len(self.source_text(event)) + (2 if selected else 0)
            if selected and size + length > policy.batch_chars:
                break
            selected.append(event)
            size += length
        return list(reversed(selected)) if policy.direction == "backward" else selected

    def _groups(self, selected: list[Event]) -> list[list[Event]]:
        texts = [self.source_text(event) for event in selected]
        units = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(texts, 1))
        data = self.llm.complete_json("boundary_detection", [
            {"role": "system", "content": (
                "你是历史记忆事件形成组件。下面每个编号是一个已经保存的完整事件，"
                "它们按经历时间连续选取，不是按相似主题聚类。复用闭合事件的形成原则，"
                "将连续编号归为一个或几个适合提炼的事件组；允许将一段时期的多件经历"
                "放在同一组，但不要编造它们的因果关系或概括成未经证明的长期偏好。"
                "每个原事件不可拆开，编号不可重复、不可改序。只输出 JSON："
                '{"events":[{"indices":[1,2]}],"residual":[3]}。'
                "没有最低字数要求；不能归入已形成组的完整编号留在 residual。"
                "不输出改写正文、情绪或摘要，正文由程序原样组装。")},
            {"role": "user", "content": units},
        ])
        # Some models return events:[[1,2],[3]] instead of the requested
        # events:[{indices:[1,2]},...]. This is an unambiguous structural variant,
        # not a semantic retry. Invalid/overlapping claims must fail, not silently
        # become "deferred" and stall an entire age window.
        raw_groups = data.get("events")
        if not isinstance(raw_groups, list):
            raise ValueError("Refinement response must contain an events list")
        normalized, owned = [], set()
        for raw_group in raw_groups:
            raw_indices = raw_group if isinstance(raw_group, list) else (
                raw_group.get("indices") if isinstance(raw_group, dict) else None)
            if not isinstance(raw_indices, list) or not raw_indices:
                raise ValueError("Refinement event must contain source indices")
            indices = normalize_indices(raw_indices)
            if (not indices or len(indices) != len(set(indices)) or owned.intersection(indices)
                    or any(i < 1 or i > len(selected) for i in indices)):
                raise ValueError("Invalid or overlapping refinement source indices")
            owned.update(indices)
            normalized.append({"indices": indices})
        residual = data.get("residual", [])
        if not isinstance(residual, list):
            raise ValueError("Refinement residual must be a list")
        residual_indices = []
        for item in residual:
            raw = item.get("indices") if isinstance(item, dict) else item if isinstance(item, list) else [item]
            if not isinstance(raw, list):
                raise ValueError("Invalid refinement residual indices")
            residual_indices.extend(normalize_indices(raw))
        if (owned.intersection(residual_indices) or len(residual_indices) != len(set(residual_indices))
                or any(i < 1 or i > len(selected) for i in residual_indices)):
            raise ValueError("Invalid or overlapping refinement residual indices")
        data = {**data, "events": normalized}
        parsed = BoundaryDetectionSkill(self.llm, self.config).parse_response(data, texts)
        groups, used = [], set()
        for fragment in parsed.completed_events:
            indices = fragment.source_indices
            if not indices or indices != list(range(min(indices), max(indices) + 1)):
                raise ValueError("Refinement groups must contain contiguous whole source events")
            if used.intersection(indices) or any(i < 1 or i > len(selected) for i in indices):
                raise ValueError("Invalid refinement source ownership")
            used.update(indices)
            groups.append([selected[i - 1] for i in indices])
        return groups

    def refine_once(self, subject_id: str, target: int, now: datetime) -> dict:
        selected = self.select(subject_id, target, now)
        if not selected:
            return {"stage": target, "status": "idle"}
        groups = self._groups(selected)
        if not groups:
            return {"stage": target, "status": "deferred",
                    "selected_ids": [event.event_id for event in selected]}
        stage = self.config.memory_refinement.stages[target - 1]
        builder = copy.copy(self.events)
        builder._config = self.config.model_copy(deep=True)
        builder._config.summary_l1_retention_ratio = stage.l1_retention_ratio
        parents, consumed, indexed = [], [], []
        try:
            for group in groups:
                content = "\n\n".join(self.source_text(event) for event in group)
                starts, ends = zip(*(time_span(event) for event in group))
                start, end = min(starts, key=utc), max(ends, key=utc)
                objects, dates, ids, attrs = {}, [], [], {}
                units, cursor = [], 0
                for event in group:
                    for name, oid in event.refinement_objects.items():
                        key = name if name not in objects or objects[name] == oid else f"{name} ({oid})"
                        objects[key] = oid
                    dates.extend(event.source_dates)
                    ids.extend(event.source_ids)
                    text = self.source_text(event)
                    unit_start, unit_end = time_span(event)
                    units.append({"event_id": event.event_id,
                        "input_level": "L0" if event.memory_stage == 0 else "L1",
                        "start_char": cursor, "end_char": cursor + len(text),
                        "event_time_start": unit_start.isoformat(), "event_time_end": unit_end.isoformat(),
                        "source_dates": event.source_dates})
                    cursor += len(text) + 2
                    for attr in event.interlocutor_attributions:
                        attrs[(attr.segment_id, attr.object_id)] = attr
                parent = builder.seal_event(content, objects=objects, subject_id=subject_id,
                    source_ids=tuple(dict.fromkeys(ids)), occurred_at=start,
                    source_dates=list(dict.fromkeys(dates)),
                    interlocutor_attributions=list(attrs.values()),
                    refinement_stage=target, persist=False,
                    seal_reason="truncated" if any(e.seal_reason != "closed" for e in group) else "closed",
                    source_context=json.dumps(units, ensure_ascii=False))
                if not parent.summaries.get("L1", "").strip():
                    raise ValueError("Refinement produced no L1; sources remain visible")
                # Keep the fresh ID assigned by EventService. Competing attempts
                # must not share vector IDs: a losing worker cleans only its own
                # unpublished points. SQL source ownership provides idempotency.
                parent.source_events = [event.event_id for event in group]
                parent.event_time_start, parent.event_time_end = start, end
                parent.create_time, parent.refined_at = now, now
                parent.refinement_policy = {"age_days": stage.age_days,
                    "l1_retention_ratio": stage.l1_retention_ratio,
                    "input_level": "L0" if target == 1 else "L1"}
                parent.refinement_source_units = units
                parents.append(parent)
                consumed.extend(group)
            # Stage vectors before publication. Until the SQL transaction commits,
            # these IDs are not active candidates and cannot be recalled.
            for parent in parents:
                indexed.append(parent)
                self.recall.index_event(parent)
            self.repo.commit_refinement(consumed, parents)
        except Exception:
            for parent in indexed:
                for key in self.recall._vector_store.list_event_logical_ids(parent.event_id):
                    self.recall._vector_store.delete(key)
            raise
        cleanup_pending = []
        for child in consumed:
            try:
                for key in self.recall._vector_store.list_event_logical_ids(child.event_id):
                    self.recall._vector_store.delete(key)
            except Exception:
                # SQL visibility is authoritative; retry cleanup via backfill.
                cleanup_pending.append(child.event_id)
        return {"stage": target, "status": "refined", "selected_ids": [e.event_id for e in selected],
                "consumed_ids": [e.event_id for e in consumed],
                "parent_ids": [e.event_id for e in parents], "cleanup_pending": cleanup_pending,
                "input_chars": sum(len(self.source_text(e)) for e in consumed),
                "L1_chars": sum(len(e.summaries["L1"]) for e in parents)}

    def run(self, subject_id: str, now: datetime | None = None) -> dict:
        if not subject_id:
            raise ValueError("A subject_id is required for refinement")
        now = now or datetime.now(timezone.utc)
        result = {"subject_id": subject_id, "as_of": now.isoformat(), "batches": [], "errors": []}
        if not self.config.memory_refinement.enabled:
            return result
        for target in range(1, len(self.config.memory_refinement.stages) + 1):
            for _ in range(self.config.memory_refinement.max_batches_per_stage):
                try:
                    batch = self.refine_once(subject_id, target, now)
                    result["batches"].append(batch)
                    if batch["status"] != "refined":
                        break
                except Exception as exc:
                    result["errors"].append({"stage": target, "error": str(exc)})
                    return result
        return result
