"""Recall query intent classification (rule-based; design/1010 §3 step 1).

第一版不调 LLM：词典子串命中 + 已知对象名匹配，输出 :class:`RecallIntent`。
对象名来自输入映射（如 {"deepseek": "OBJ-DEEPSEEK"}），由调用方提供。
"""

from __future__ import annotations

from dataclasses import dataclass

_POSITIVE = ("开心", "高兴", "快乐", "喜悦", "喜欢", "满意", "幸福", "愉快")
_NEGATIVE = ("难过", "伤心", "悲伤", "痛苦", "害怕", "恐惧", "生气", "愤怒", "焦虑", "失望")
_RECENCY = (
    "最近", "刚才", "刚刚", "今天", "昨天", "上一轮", "上轮",
    "前几轮", "最近一次", "近期", "这几次",
)


@dataclass
class RecallIntent:
    """结构化查询意图（规则版）。"""

    object_id: str | None = None    # 查询指向的对象（命中 known_objects）
    emotion: str | None = None      # "positive" / "negative" / None
    recency: bool = False           # 查询带近因信号（最近 / 刚才 / …）
    fact: bool = False              # 事实型查询（预留，供后续混合）


class RuleIntentClassifier:
    """词典/子串规则意图分类器（不调用 LLM）。"""

    def __init__(
        self,
        *,
        entity_lexicon: list[str] | None = None,
        emotion_lexicon: list[str] | None = None,
        fact_lexicon: list[str] | None = None,
        known_objects: dict[str, str] | None = None,
    ):
        self._entity_lexicon = list(entity_lexicon or [])
        self._emotion_lexicon = list(emotion_lexicon or [])
        self._fact_lexicon = list(fact_lexicon or [])
        self._known_objects = dict(known_objects or {})

    def classify(self, query: str) -> RecallIntent:
        q = (query or "").strip()
        if not q:
            return RecallIntent()
        return RecallIntent(
            object_id=self._match_object(q),
            emotion=self._match_emotion(q),
            recency=any(w in q for w in _RECENCY),
            fact=any(w in q for w in self._fact_lexicon) if self._fact_lexicon else False,
        )

    def _match_object(self, q: str) -> str | None:
        # 长名优先，避免短名误匹配
        for name in sorted(self._known_objects, key=len, reverse=True):
            if name and name in q:
                return self._known_objects[name]
        return None

    def _match_emotion(self, q: str) -> str | None:
        pos = any(w in q for w in _POSITIVE) or any(
            w in q for w in self._emotion_lexicon if w in _POSITIVE
        )
        neg = any(w in q for w in _NEGATIVE) or any(
            w in q for w in self._emotion_lexicon if w in _NEGATIVE
        )
        if pos and not neg:
            return "positive"
        if neg and not pos:
            return "negative"
        return None
