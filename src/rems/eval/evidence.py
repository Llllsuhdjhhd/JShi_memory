"""Retrieval diagnostics, separate from answer and delivered-memory grading.

Source IDs use the existing ``sample_id:dia_id`` namespace. Session/chunk hits
are provenance bounds; turn hits require actual original text, never a source-ID
fallback. These metrics describe retrieved events, not what the core finally saw.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from .dataset import sessions, turn_text


def _norm(text):
    # The real JShi ledger renders speaker prefixes with a fullwidth colon.
    text = re.sub(r":\s*", ": ", text.casefold().replace("：", ":"))
    return " ".join(text.split())


def retrieval_evidence(sample, events, evidence, *, tolerance=timedelta(minutes=5)):
    timeline = sessions(sample)
    sid = sample["sample_id"]
    turns = {t["dia_id"]: (index, t) for index, _, items in timeline for t in items}
    gold = {d for item in evidence for d in re.findall(r"\bD\d+:\d+\b", str(item))}
    gold_sessions = {int(d[1:].split(":")[0]) for d in gold}
    got_sessions, got_chunks, got_turns = set(), set(), set()
    mappings = []
    for event in events:
        sources = event.get("source_ids") or []
        local = {s[len(sid)+1:] for s in sources if s.startswith(sid + ":")}
        chunk = local & turns.keys()
        foreign = [s for s in sources if re.search(r":D\d+:\d+$", s)
                   and not s.startswith(sid + ":")]
        matched_sessions = {turns[d][0] for d in chunk}
        method = "source_ids" if chunk else "unmatched"
        # Explicit sources take precedence. Never relabel another sample's
        # event by a coincident timestamp; bare or unknown sources stay unknown.
        if not sources and event.get("occurred_at"):
            occurred = datetime.fromisoformat(event["occurred_at"])
            distances = [(abs(occurred-date), index) for index, date, _ in timeline
                         if occurred.utcoffset() == date.utcoffset()]
            if distances:
                nearest = min(distance for distance, _ in distances)
                closest = [index for distance, index in distances if distance == nearest]
                if nearest <= tolerance and len(closest) == 1:
                    matched_sessions = set(closest)
                    method = "time"
        raw = _norm(event.get("content_raw") or "")
        exact = set()
        if raw:
            for dia, (index, turn) in turns.items():
                if index not in matched_sessions or (sources and dia not in chunk):
                    continue
                # Require a complete attributed turn, rather than an arbitrary
                # sentence or an unattributed generic reply.
                line = _norm(f"{turn['speaker']}: {turn_text(turn)}")
                if line in raw:
                    exact.add(dia)
        got_sessions.update(matched_sessions)
        got_chunks.update(chunk)
        got_turns.update(exact)
        mappings.append({"event_id": event["event_id"], "method": method,
                         "sessions": sorted(matched_sessions), "chunk_dias": sorted(chunk),
                         "turn_dias": sorted(exact), "foreign_source_ids": foreign})
    result = {"scope": "retrieved_events", "evidence": sorted(gold), "events": mappings,
              "cross_conversation_events": sum(bool(m["foreign_source_ids"]) for m in mappings),
              "unmatched_events": sum(m["method"] == "unmatched" for m in mappings)}
    for level, expected, found in (("session", gold_sessions, got_sessions),
                                    ("chunk", gold, got_chunks), ("turn", gold, got_turns)):
        result[level + "_hit"] = bool(expected & found) if expected else None
        result[level + "_recall"] = len(expected & found) / len(expected) if expected else None
    return result


def candidate_ranks(sample, candidates, events, evidence):
    """gold 证据在候选序列里的排名与去向（诊断用；不参与评分）。

    ``candidates`` 是 ``recall_candidates`` 记录列表，``events`` 是
    ``{event_id: {"source_ids": [...]}}``。同一事件可能以多种材料类型出现，因此按
    ``(call, rank)`` 取最靠前的一次。

    ``recalled=False`` 有两种含义，必须区分：真的没进候选池，或候选在条数上限处被截断。
    后者由 ``truncated`` 标出。要拿到完整排名，recall 时不要传 ``limit``，
    这样预算放不下的候选会以 ``no_room`` 继续记录到最后。
    """
    sid = sample["sample_id"]
    gold = {d for item in evidence for d in re.findall(r"\bD\d+:\d+\b", str(item))}
    hits = []
    for row in candidates or []:
        event = events.get(row.get("event_id"))
        if not event:
            continue
        local = {s[len(sid) + 1:] for s in event.get("source_ids") or []
                 if s.startswith(sid + ":")}
        if gold & local:
            hits.append(row)
    truncated = any(row.get("reason") == "limit" for row in candidates or [])
    if not hits:
        return {"recalled": False, "gold": sorted(gold),
                "candidates": len(candidates or []), "truncated": truncated}
    best = min(hits, key=lambda row: (row.get("call", 0), row["rank"]))
    selected = [row for row in hits if row.get("selected")]
    return {
        "recalled": True,
        "gold": sorted(gold),
        "candidates": len(candidates or []),
        "best_rank": best["rank"],
        "best_call": best.get("call", 0),
        "best_reason": best.get("reason"),
        "best_selected": bool(best.get("selected")),
        "best_level": best.get("representation_level"),
        "best_chars": best.get("chars", 0),
        "selected_rank": min((row["rank"] for row in selected), default=None),
        "truncated": truncated,
    }
