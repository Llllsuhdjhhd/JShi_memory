from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import REMSConfig
from ..models.event import Event, Importance
from .act_text import resolve_event_act_text, resolve_query_act_text

if TYPE_CHECKING:
    from ..storage.repository import EventRepository, RoleRepository


@dataclass(frozen=True)
class TriBandVectors:
    vector_act: list[float]
    vector_emo: list[float]
    vector_ent: list[float]


class _HashEmbedder:
    def __init__(self, dim: int):
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        seed = hashlib.sha256((text or "").encode("utf-8")).digest()
        values: list[float] = []
        counter = 0
        while len(values) < self._dim:
            block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
            for b in block:
                values.append((b / 255.0) * 2.0 - 1.0)
                if len(values) >= self._dim:
                    break
            counter += 1
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class TriBandEncoder:
    """Encode events into act / emo / ent vector bands (§4.4.0)."""

    _IMPORTANCE_ORDER = {
        Importance.S: 5,
        Importance.A: 4,
        Importance.B: 3,
        Importance.C: 2,
        Importance.D: 1,
    }

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository | None = None,
        role_repo: RoleRepository | None = None,
        text_embedder: Any | None = None,
    ):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        tb = config.tri_band
        self._dim_act = tb.vector_dim_act
        self._dim_emo = tb.vector_dim_emo
        self._dim_ent = tb.vector_dim_ent
        if text_embedder is not None:
            self._text = text_embedder
        elif (config.embedding.provider or "local").lower() == "hash":
            self._text = _HashEmbedder(self._dim_act)
        else:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(config.embedding.model_name)

                class _STWrapper:
                    def embed_query(self, text: str) -> list[list[float]]:
                        return [model.encode(text, normalize_embeddings=True).tolist()]

                self._text = _STWrapper()
            except Exception:
                self._text = _HashEmbedder(self._dim_act)

    def _embed_text(self, text: str, dim: int) -> list[float]:
        if hasattr(self._text, "embed_query"):
            raw = self._text.embed_query(text)[0]
        elif hasattr(self._text, "embed"):
            raw = self._text.embed(text)
        else:
            raw = self._text([text])[0]
        return self._project(raw, dim)

    @staticmethod
    def _project(vec: list[float], dim: int) -> list[float]:
        if len(vec) == dim:
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            return [v / norm for v in vec]
        if len(vec) > dim:
            truncated = vec[:dim]
            norm = math.sqrt(sum(v * v for v in truncated)) or 1.0
            return [v / norm for v in truncated]
        out = list(vec)
        while len(out) < dim:
            out.append(0.0)
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def _emo_vector(self, event: Event) -> list[float]:
        dims = 8
        if not event.role_list:
            scalars = [0.0] * (3 + dims)
        else:
            emos = [entry.emotional_model.emotion for entry in event.role_list]
            anger = max(e.anger for e in emos)
            fear = max(e.fear for e in emos)
            joy = max(e.joy for e in emos)
            sadness = max(e.sadness for e in emos)
            surprise = max(e.surprise for e in emos)
            disgust = max(e.disgust for e in emos)
            trust = max(e.trust for e in emos)
            anticipation = max(e.anticipation for e in emos)
            valence = sum(entry.emotional_model.valence for entry in event.role_list) / len(event.role_list)
            arousal = max(entry.emotional_model.arousal for entry in event.role_list)
            energy = sum(entry.emotional_model.energy for entry in event.role_list)
            scalars = [valence, arousal, energy, anger, fear, joy, sadness, surprise, disgust, trust, anticipation]
        # Fixed projection matrix from scalars to emo dim (deterministic, no train step).
        out = [0.0] * self._dim_emo
        for i, s in enumerate(scalars):
            for j in range(self._dim_emo):
                phase = math.sin((i + 1) * (j + 1) * 0.17) * s
                out[j] += phase
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]

    def _append_entity_parts(self, parts: list[str], role_id: str) -> None:
        if not self._config.tri_band.ent_names_only:
            parts.append(role_id)
        if self._role_repo:
            role = self._role_repo.get(role_id)
            if role:
                if role.name:
                    parts.append(role.name)
                parts.extend(a for a in (role.aliases or []) if a)

    def _entity_text(self, event: Event) -> str:
        parts: list[str] = []
        if event.role_list:
            for entry in event.role_list:
                self._append_entity_parts(parts, entry.role_id)
            return " ".join(dict.fromkeys(parts))
        if event.is_abstract and event.source_events and self._event_repo:
            leaf_ids = self._event_repo.resolve_basic_event_ids(event.event_id)
            for eid in leaf_ids:
                leaf = self._event_repo.get(eid)
                if leaf and leaf.role_list:
                    for entry in leaf.role_list:
                        self._append_entity_parts(parts, entry.role_id)
        return " ".join(dict.fromkeys(parts)) if parts else event.event_id

    def _act_text(self, event: Event) -> str:
        return resolve_event_act_text(event, self._config.tri_band)

    def encode(self, event: Event) -> TriBandVectors:
        act = self._embed_text(self._act_text(event), self._dim_act)
        emo = self._emo_vector(event)
        ent = self._embed_text(self._entity_text(event), self._dim_ent)
        return TriBandVectors(vector_act=act, vector_emo=emo, vector_ent=ent)

    def encode_query(
        self,
        query: str,
        *,
        valence: float = 0.0,
        arousal: float = 0.0,
        entity_hint: str = "",
        act_source: str | None = None,
    ) -> TriBandVectors:
        act_text = act_source if act_source is not None else resolve_query_act_text(
            query, self._config.tri_band
        )
        act = self._embed_text(act_text, self._dim_act)
        fake_event = Event(
            event_id="QRY",
            content_raw=query,
            role_list=[],
        )
        fake_event.role_list = []
        emo = self._emo_vector(fake_event)
        if arousal > 0 or valence != 0:
            emo = self._emo_vector_from_scalars(valence, arousal)
        ent_text = entity_hint or query
        ent = self._embed_text(ent_text, self._dim_ent)
        return TriBandVectors(vector_act=act, vector_emo=emo, vector_ent=ent)

    def _emo_vector_from_scalars(self, valence: float, arousal: float) -> list[float]:
        out = [0.0] * self._dim_emo
        for j in range(self._dim_emo):
            out[j] = math.sin((j + 1) * 0.11) * valence + math.cos((j + 1) * 0.13) * arousal
        norm = math.sqrt(sum(v * v for v in out)) or 1.0
        return [v / norm for v in out]
