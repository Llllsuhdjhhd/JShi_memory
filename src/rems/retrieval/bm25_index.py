from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.event import Event
    from ..storage.repository import RoleRepository


def _tokenize(text: str) -> list[str]:
    """Character-level tokens for classical Chinese / mixed prose."""
    return [c for c in (text or "") if not c.isspace()]


def _event_index_text(event: Event, role_repo: RoleRepository | None) -> str:
    parts: list[str] = []
    summaries = event.summaries or {}
    for key in sorted(summaries.keys(), key=lambda x: (len(x), x)):
        parts.append(summaries[key])
    if not parts and event.content_raw:
        parts.append(event.content_raw[:400])
    for entry in event.role_list:
        if role_repo is not None:
            role = role_repo.get(entry.role_id)
            if role is not None:
                if role.name:
                    parts.append(role.name)
                parts.extend(a for a in (role.aliases or []) if a)
        parts.append(entry.role_id)
    keywords = getattr(event, "keywords", None) or []
    if keywords:
        parts.extend(keywords)
    location = getattr(event, "location", None)
    if location:
        parts.append(str(location))
    return " ".join(parts)


class EventBm25Index:
    """In-memory BM25 over sealed events (L2 summaries + role names + optional keywords)."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._event_ids: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._id_to_pos: dict[str, int] = {}
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._role_repo: RoleRepository | None = None

    def set_role_repo(self, role_repo: RoleRepository) -> None:
        self._role_repo = role_repo

    def clear(self) -> None:
        self._event_ids.clear()
        self._doc_tokens.clear()
        self._id_to_pos.clear()
        self._df.clear()
        self._avgdl = 0.0

    def rebuild(self, events: list[Event]) -> None:
        self.clear()
        for ev in events:
            self.upsert(ev)

    def upsert(self, event: Event) -> None:
        tokens = _tokenize(_event_index_text(event, self._role_repo))
        eid = event.event_id
        if eid in self._id_to_pos:
            self.remove(eid)
        pos = len(self._event_ids)
        self._id_to_pos[eid] = pos
        self._event_ids.append(eid)
        self._doc_tokens.append(tokens)
        seen: set[str] = set()
        for t in tokens:
            if t not in seen:
                self._df[t] = self._df.get(t, 0) + 1
                seen.add(t)
        n = len(self._doc_tokens)
        self._avgdl = sum(len(d) for d in self._doc_tokens) / n if n else 0.0

    def remove(self, event_id: str) -> None:
        pos = self._id_to_pos.pop(event_id, None)
        if pos is None:
            return
        old_tokens = self._doc_tokens[pos]
        seen: set[str] = set()
        for t in old_tokens:
            if t not in seen:
                c = self._df.get(t, 0) - 1
                if c <= 0:
                    self._df.pop(t, None)
                else:
                    self._df[t] = c
                seen.add(t)
        last = len(self._event_ids) - 1
        if pos != last:
            moved_id = self._event_ids[last]
            self._event_ids[pos] = moved_id
            self._doc_tokens[pos] = self._doc_tokens[last]
            self._id_to_pos[moved_id] = pos
        self._event_ids.pop()
        self._doc_tokens.pop()
        n = len(self._doc_tokens)
        self._avgdl = sum(len(d) for d in self._doc_tokens) / n if n else 0.0

    def search(
        self,
        query: str,
        *,
        top_k: int = 40,
        eligible_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        q_tokens = _tokenize(query)
        if not q_tokens or not self._doc_tokens:
            return []
        n = len(self._doc_tokens)
        scores: list[tuple[str, float]] = []
        for i, eid in enumerate(self._event_ids):
            if eligible_ids is not None and eid not in eligible_ids:
                continue
            doc = self._doc_tokens[i]
            dl = len(doc)
            if dl == 0:
                continue
            score = 0.0
            for qt in q_tokens:
                df = self._df.get(qt, 0)
                if df == 0:
                    continue
                tf = doc.count(qt)
                if tf == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                denom = tf + self._k1 * (1.0 - self._b + self._b * dl / max(self._avgdl, 1.0))
                score += idf * (tf * (self._k1 + 1.0)) / denom
            if score > 0:
                scores.append((eid, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
