from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..config import REMSConfig

# 意图路由：把查询映射为 tri-band（act/emo/ent）权重。
# 软混合（默认）：按词典（可选原型向量）信号算 entity/emotion/fact 混合系数，对三组配置权重做凸组合，
# 天然支持混合意图（如"他当时害怕谁"= 实体 + 情绪）。可回退到旧的正则硬三选一（§4.4.0 v1）。


@dataclass(frozen=True)
class RecallIntent:
    weights: tuple[float, float, float]
    label: str


# 旧版正则——仅用于硬选回退分支。
_ENTITY_PATTERNS = re.compile(
    r"谁|人物|角色|轨迹|做过什么|干什么|where|who|character|person",
    re.I,
)
_EMO_PATTERNS = re.compile(
    r"感觉|心情|情绪|害怕|开心|难过|feel|emotion|mood|afraid|happy|sad",
    re.I,
)


def _count_hits(text: str, lexicon: list[str]) -> int:
    """Count case-insensitive substring hits of *lexicon* terms in *text*."""
    if not text or not lexicon:
        return 0
    low = text.lower()
    return sum(1 for w in lexicon if w and w.lower() in low)


class RecallIntentClassifier:
    """Intent router producing tri-band weights.

    两种模式：
    - 软混合（``recall_intent_soft_blend_enabled=True``，默认）：词典信号（可选叠加原型向量
      余弦）→ softmax 得到 entity/emotion/fact 混合系数 α → 对 ``tri_band`` 三组权重做凸组合。
      无任何信号时 fact 基线占优，结果回退到 ``weight_fact``。
    - 硬选 dominant（``=False``）：旧的正则三选一，便于 A/B 对照。

    ``embedder`` 可选（通常注入 ``TriBandEncoder``），仅当
    ``recall_intent_prototype_enabled=True`` 时用于计算原型向量信号。
    """

    def __init__(self, config: REMSConfig, embedder: Any | None = None):
        self._config = config
        self._embedder = embedder
        # 原型向量缓存：intent → act 向量（懒构造）。
        self._prototype_cache: dict[str, list[float]] | None = None

    def classify(self, query: str) -> RecallIntent:
        if not self._config.recall_intent_soft_blend_enabled:
            return self._classify_hard(query)

        tb = self._config.tri_band
        scores = self._signal_scores(query or "")
        alpha = self._softmax(scores, self._config.recall_intent_softmax_temperature)

        w_ent = tb.weight_entity
        w_emo = tb.weight_emotion
        w_fact = tb.weight_fact
        a_ent, a_emo, a_fact = alpha["entity"], alpha["emotion"], alpha["fact"]
        blended = tuple(
            a_ent * w_ent[i] + a_emo * w_emo[i] + a_fact * w_fact[i]
            for i in range(3)
        )
        if not self._config.tri_band.emo_search_enabled:
            blended = (blended[0], 0.0, blended[2])
        # 防御性归一（三组权重各自和为 1，系数和为 1 时结果亦和为 1；此处兜底数值误差）。
        total = sum(blended) or 1.0
        blended = tuple(v / total for v in blended)

        dominant = max(alpha, key=lambda k: alpha[k])
        return RecallIntent(weights=blended, label=f"blend:{dominant}")

    # ------------------------------------------------------------------
    def _classify_hard(self, query: str) -> RecallIntent:
        tb = self._config.tri_band
        if _ENTITY_PATTERNS.search(query or ""):
            w = tb.weight_entity
        elif _EMO_PATTERNS.search(query or ""):
            w = tb.weight_emotion
        else:
            w = tb.weight_fact
        if not tb.emo_search_enabled:
            w = (w[0], 0.0, w[2])
            total = w[0] + w[2] or 1.0
            w = (w[0] / total, 0.0, w[2] / total)
        label = "entity_trace" if _ENTITY_PATTERNS.search(query or "") else (
            "emotion" if _EMO_PATTERNS.search(query or "") else "fact"
        )
        return RecallIntent(weights=w, label=label)

    # ------------------------------------------------------------------
    def _signal_scores(self, query: str) -> dict[str, float]:
        cfg = self._config
        scores = {
            "entity": float(_count_hits(query, cfg.recall_intent_entity_lexicon)),
            "emotion": float(_count_hits(query, cfg.recall_intent_emotion_lexicon)),
            # fact 作为残余类：基线 + 自身词典命中，保证无信号时回退到 weight_fact。
            "fact": float(cfg.recall_intent_fact_baseline)
            + float(_count_hits(query, cfg.recall_intent_fact_lexicon)),
        }
        if cfg.recall_intent_prototype_enabled and self._embedder is not None:
            proto = self._prototype_signal(query)
            w = cfg.recall_intent_prototype_weight
            for key in scores:
                scores[key] += w * proto.get(key, 0.0)
        return scores

    def _prototype_signal(self, query: str) -> dict[str, float]:
        """Cosine similarity of the query act-vector to each intent prototype.

        原型向量由各类词典拼接成种子短语后嵌入得到（懒构造并缓存）。负相似度截断为 0，
        避免把"明显不属于该类"误当成负贡献干扰 softmax。
        """
        try:
            protos = self._ensure_prototypes()
            q_vec = self._embed_act(query)
            if not q_vec:
                return {}
            return {k: max(0.0, _cosine(q_vec, v)) for k, v in protos.items()}
        except Exception:  # noqa: BLE001 — 原型信号是可选增强，失败时静默退回词典信号。
            return {}

    def _ensure_prototypes(self) -> dict[str, list[float]]:
        if self._prototype_cache is not None:
            return self._prototype_cache
        cfg = self._config
        seeds = {
            "entity": " ".join(cfg.recall_intent_entity_lexicon),
            "emotion": " ".join(cfg.recall_intent_emotion_lexicon),
            "fact": " ".join(cfg.recall_intent_fact_lexicon),
        }
        self._prototype_cache = {
            k: self._embed_act(text) for k, text in seeds.items() if text
        }
        return self._prototype_cache

    def _embed_act(self, text: str) -> list[float]:
        """Get the act-band vector for *text* via the injected encoder."""
        if not text:
            return []
        vec = self._embedder.encode_query(text)
        return list(getattr(vec, "vector_act", []) or [])

    @staticmethod
    def _softmax(scores: dict[str, float], temperature: float) -> dict[str, float]:
        t = temperature if temperature and temperature > 0 else 1.0
        keys = list(scores.keys())
        vals = [scores[k] / t for k in keys]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        ssum = sum(exps) or 1.0
        return {k: e / ssum for k, e in zip(keys, exps)}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
