"""LoCoMo 评测适配层（纯函数部分；运行器见 scripts/locomo_eval.py）。

- 会话 → MemoryBatch：每个 session 一批，一轮一行 ``Speaker: text``；
- 答案打分：LoCoMo 风格 token F1（按 category 处理）；
- 时间反查：召回事件 occurred_at → session（时间只能定位到 session），
  source_ids ∩ 该 session dia_id → dia 级定位；与 evidence 比对命中。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..port import MemoryBatch, MemoryExperience

CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def parse_locomo_datetime(raw: str) -> datetime:
    """``'1:56 pm on 8 May, 2023'`` → datetime(2023, 5, 8, 13, 56)。"""
    s = " ".join(raw.strip().split())
    return datetime.strptime(s, "%I:%M %p on %d %B, %Y")


@dataclass
class Session:
    sample_id: str
    index: int
    occurred_at: datetime
    raw_date_time: str
    turns: list[dict[str, Any]]

    @property
    def session_id(self) -> str:
        return f"{self.sample_id}_session_{self.index}"

    @property
    def dia_ids(self) -> list[str]:
        return [t["dia_id"] for t in self.turns]


def iter_sessions(sample: dict[str, Any]) -> list[Session]:
    conv = sample["conversation"]
    out: list[Session] = []
    for key, turns in conv.items():
        m = _SESSION_KEY.match(key)
        if not m or not isinstance(turns, list) or not turns:
            continue
        n = int(m.group(1))
        raw = conv.get(f"session_{n}_date_time") or ""
        out.append(Session(sample["sample_id"], n, parse_locomo_datetime(raw), raw, turns))
    out.sort(key=lambda s: s.index)
    return out


def turn_line(turn: dict[str, Any], *, include_image_caption: bool = True) -> str:
    text = (turn.get("text") or "").strip()
    cap = (turn.get("blip_caption") or "").strip()
    if include_image_caption and cap:
        text = f"{text} [shares an image: {cap}]".strip()
    return f"{turn['speaker']}: {text}"


PROTAGONIST = "protagonist"


def speaker_objects(sample: dict[str, Any]) -> dict[str, str]:
    """对象映射：speaker_a 即主体（``protagonist``，不进对象表）；speaker_b 按名字一个对象。

    对象 id 带 ``{sample_id}:`` 前缀，保证跨对话同名不串，并可从事件反推来源对话。
    """
    conv = sample["conversation"]
    b = conv.get("speaker_b")
    return {b: f"{sample['sample_id']}:{b}"} if b else {}


def chunk_session(session: Session, *, soft_chars: int = 3000,
                  include_image_caption: bool = True) -> list[list[dict[str, Any]]]:
    """写侧软触发：累计到 ~soft_chars 后在当前轮次（完整话语）结束处切；不硬切、不跨 session。

    单轮超长时整轮保留（宁长不断句）。
    """
    chunks: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    size = 0
    for t in session.turns:
        line = turn_line(t, include_image_caption=include_image_caption)
        cur.append(t)
        size += len(line) + 1
        if size >= soft_chars:
            chunks.append(cur)
            cur, size = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def session_batches(
    sample: dict[str, Any], session: Session, *, subject_id: str,
    soft_chars: int = 3000, include_image_caption: bool = True,
) -> list[MemoryBatch]:
    """一个 session → 若干批（软触发切块）；每批一条经历：一轮一行「Speaker: text」。"""
    out: list[MemoryBatch] = []
    chunks = chunk_session(session, soft_chars=soft_chars,
                           include_image_caption=include_image_caption)
    for k, turns in enumerate(chunks, 1):
        seg = session.session_id if len(chunks) == 1 else f"{session.session_id}_part_{k}"
        exp = MemoryExperience(
            subject_id=subject_id,
            text="\n".join(turn_line(t, include_image_caption=include_image_caption) for t in turns),
            objects=speaker_objects(sample),
            source_ids=tuple(t["dia_id"] for t in turns),
            segment_id=seg,
            occurred_at=session.occurred_at,
        )
        out.append(MemoryBatch(subject_id=subject_id, experiences=(exp,)))
    return out


def session_batch(
    sample: dict[str, Any], session: Session, *, include_image_caption: bool = True,
) -> MemoryBatch:
    """整 session 一批（不切块）；subject_id = sample_id。"""
    return session_batches(sample, session, subject_id=sample["sample_id"],
                           soft_chars=10**9, include_image_caption=include_image_caption)[0]


def sample_of_segment(segment_id: str | None) -> str | None:
    """``conv-26_session_3[_part_2]`` → ``conv-26``。"""
    if not segment_id:
        return None
    m = re.match(r"^(.*?)_session_\d+", segment_id)
    return m.group(1) if m else None


_SENT_END = re.compile(r"[.!?。！？…](?:[\"')\]]*)(?=\s|$)|\n")


def cap_context(text: str, max_chars: int = 3000) -> str:
    """把回忆上下文截到 ≤ max_chars，回退到最近的完整句子边界（找不到才按字截）。"""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    ends = [m.end() for m in _SENT_END.finditer(head)]
    return head[: ends[-1]].rstrip() if ends else head


# ----------------------------------------------------------------------
# 时间反查
# ----------------------------------------------------------------------


@dataclass
class EventMapping:
    event_id: str
    sample_id: str | None = None  # 来源对话（由 segment_id 解析）
    cross_conversation: bool = False
    occurred_at: str | None = None
    session: int | None = None  # 由时间定位
    session_delta_s: float | None = None
    dia_ids: list[str] = field(default_factory=list)  # 由 source_ids ∩ session
    source_ids: list[str] = field(default_factory=list)
    matched: bool = False
    turn_dia_ids: list[str] = field(default_factory=list)  # 原文精确定位
    turn_method: str | None = None


def _naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def map_event_to_session(
    sessions: list[Session],
    event_id: str,
    occurred_at: datetime | None,
    source_ids: Iterable[str],
    *,
    tolerance: timedelta = timedelta(minutes=5),
    sample_id: str | None = None,
    content_raw: str | None = None,
) -> EventMapping:
    src = list(source_ids or [])
    m = EventMapping(event_id, sample_id, False, occurred_at.isoformat() if occurred_at else None, None, None, [], src)
    if occurred_at is None:
        return m
    t = _naive(occurred_at)
    best: tuple[float, Session] | None = None
    for s in sessions:
        if sample_id is not None and s.sample_id != sample_id:
            continue
        d = abs((t - s.occurred_at).total_seconds())
        if d <= tolerance.total_seconds() and (best is None or d < best[0]):
            best = (d, s)
    if best is None:
        return m
    m.session, m.session_delta_s, m.matched = best[1].index, best[0], True
    srcset = set(src)
    m.dia_ids = [d for d in best[1].dia_ids if d in srcset]
    if content_raw is not None:
        m.turn_dia_ids, m.turn_method = locate_turns(best[1], content_raw, src)
    return m


def evidence_sessions(evidence: Iterable[str]) -> set[int]:
    out: set[int] = set()
    for e in evidence or []:
        for part in re.split(r"[;,\s]+", str(e)):
            mm = re.match(r"D(\d+):\d+", part.strip())
            if mm:
                out.add(int(mm.group(1)))
    return out


def evidence_dias(evidence: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for e in evidence or []:
        for part in re.split(r"[;,\s]+", str(e)):
            if re.match(r"D\d+:\d+$", part.strip()):
                out.add(part.strip())
    return out


def _tnorm(s: str) -> str:
    return " ".join(str(s).lower().split())


_SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")


def locate_turns(
    session: Session, content_raw: str, source_ids: Iterable[str], *,
    include_image_caption: bool = True, min_sentence_chars: int = 12,
) -> tuple[list[str], str]:
    """在 session 内把事件原文精确定位到 dia_id。

    1) 每行「Speaker: text」规范化后是事件原文的子串 → 命中；
    2) 否则该行任一句（≥min_sentence_chars）是子串 → 命中；
    3) 都没有 → 回退到 source_ids ∩ session。返回 (dia_ids, method)。
    """
    body = _tnorm(content_raw or "")
    src = set(source_ids or [])
    if body:
        hits = [t["dia_id"] for t in session.turns
                if _tnorm(turn_line(t, include_image_caption=include_image_caption)) in body]
        if hits:
            return hits, "line"
        hits = []
        for t in session.turns:
            sents = [x for x in _SENT_SPLIT.split((t.get("text") or "").strip())
                     if len(x.strip()) >= min_sentence_chars]
            if any(_tnorm(x) in body for x in sents):
                hits.append(t["dia_id"])
        if hits:
            return hits, "sentence"
    return [d for d in session.dia_ids if d in src], "source_ids"


def evidence_hits(mappings: list[EventMapping], evidence: Iterable[str]) -> dict[str, Any]:
    """三档命中：session（时间定位）、chunk（批 source_ids）、turn（原文精确定位 dia_id）。"""
    ev_s, ev_d = evidence_sessions(evidence), evidence_dias(evidence)
    got_s = {m.session for m in mappings if m.session is not None}
    got_c = {d for m in mappings for d in m.dia_ids}
    got_t = {d for m in mappings for d in (m.turn_dia_ids or [])}
    return {
        "evidence": sorted(ev_d),
        "session_hit": bool(ev_s & got_s),
        "chunk_hit": bool(ev_d & got_c),
        "turn_hit": bool(ev_d & got_t),
        "dia_hit": bool(ev_d & got_c),  # 兼容旧字段 = chunk_hit
        "session_recall": (len(ev_s & got_s) / len(ev_s)) if ev_s else None,
        "chunk_recall": (len(ev_d & got_c) / len(ev_d)) if ev_d else None,
        "turn_recall": (len(ev_d & got_t) / len(ev_d)) if ev_d else None,
        "dia_recall": (len(ev_d & got_c) / len(ev_d)) if ev_d else None,
    }


# ----------------------------------------------------------------------
# 打分（参照 snap-research/locomo task_eval/evaluation.py，未做词干化）
# ----------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    s = str(s).replace(",", "")
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(ground_truth).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    preds = [x.strip() for x in str(prediction).split(",") if x.strip()]
    gts = [x.strip() for x in str(ground_truth).split(",") if x.strip()]
    if not preds or not gts:
        return 0.0
    return sum(max(token_f1(p, g) for p in preds) for g in gts) / len(gts)


ADVERSARIAL_MARKERS = ("no information available", "not mentioned")


def score_answer(category: int, prediction: str, answer: Any) -> float:
    if category == 5:
        low = str(prediction).lower()
        return 1.0 if any(m in low for m in ADVERSARIAL_MARKERS) else 0.0
    ans = str(answer)
    if category == 3:
        ans = ans.split(";")[0].strip()
    if category == 1:
        return multi_answer_f1(prediction, ans)
    return token_f1(prediction, ans)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, list[float]] = {}
    for r in records:
        by.setdefault(CATEGORY_NAMES.get(r["category"], str(r["category"])), []).append(r["f1"])
    out: dict[str, Any] = {
        "n": len(records),
        "f1_overall": (sum(r["f1"] for r in records) / len(records)) if records else 0.0,
        "f1_by_category": {k: {"n": len(v), "f1": sum(v) / len(v)} for k, v in sorted(by.items())},
    }
    with_ev = [r for r in records if r.get("evidence_hits", {}).get("evidence")]
    if with_ev:
        out["evidence"] = {
            "n_with_evidence": len(with_ev),
            **{f"{lv}_hit_rate": sum(r["evidence_hits"][f"{lv}_hit"] for r in with_ev) / len(with_ev)
               for lv in ("session", "chunk", "turn")},
            **{f"{lv}_recall_mean": sum(r["evidence_hits"][f"{lv}_recall"] or 0 for r in with_ev) / len(with_ev)
               for lv in ("session", "chunk", "turn")},
        }
    judged = [r for r in records if r.get("judge") is not None]
    if judged:
        out["llm_judge_acc"] = sum(1 for r in judged if r["judge"]) / len(judged)
    out["unmatched_events"] = sum(r.get("unmatched_events", 0) for r in records)
    tot = sum(len(r.get("recalled", [])) for r in records)
    cross = sum(1 for r in records for m in r.get("recalled", []) if m.get("cross_conversation"))
    out["recalled_events_total"] = tot
    out["cross_conversation_ratio"] = (cross / tot) if tot else 0.0
    return out


# ----------------------------------------------------------------------
# 答题提示
# ----------------------------------------------------------------------

ANSWER_STYLES = ("strict", "english_lenient")

_ANSWER_HEAD = (
    "You are answering a question about a long-term conversation between {a} and {b}, "
    "using only the retrieved memories below. Each memory has the date it happened.\n"
    "Answer with a short phrase (a few words), no explanation. For time questions, give "
    "an absolute date/month/year resolved from the memory date where possible. "
)
_ANSWER_TAIL = {
    "strict": "If the memories do not contain the answer, reply exactly: "
              "Not mentioned in the conversation.\n\n",
    "english_lenient": "Always answer in English, even if the memories are written in another "
                       "language. Give your best answer from any relevant evidence in the memories; "
                       "reply \"Not mentioned in the conversation\" only when the memories contain "
                       "truly no evidence related to the question.\n\n",
}


def build_answer_prompt(style: str, *, a: str, b: str, memories: str, question: str) -> str:
    if style not in ANSWER_STYLES:
        raise ValueError(f"answer style must be one of {ANSWER_STYLES}")
    return (_ANSWER_HEAD.format(a=a, b=b) + _ANSWER_TAIL[style]
            + f"Memories:\n{memories}\n\nQuestion: {question}\nShort answer:")
