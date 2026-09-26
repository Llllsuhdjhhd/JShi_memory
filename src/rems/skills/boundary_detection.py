from __future__ import annotations

import logging
from typing import Any, Optional

# 边界检测技能：残影 + 当前输入 + 未完成库 → 已闭环片段、剩余残影、新未完成条目。
# Prompt 内嵌白皮书 1.1.7 防碎片化聚合原则（同一段落内琐碎动作合并为一条基本事件）。
# 新增（2026-05）：80/20 强制分裂 —— 过长未完成事件必须在逻辑闭环点切成
#   一条 ``completed_events`` 前缀 + 一条 ``new_unclosed`` 尾部，二者用 ``split_id`` 配对。

from pydantic import BaseModel, Field

from ..config import REMSConfig
from ..llm.provider import LLMProvider
from ..llm.prompts import BOUNDARY_SYSTEM, BOUNDARY_USER, build_user_mode_block
from ..models.metabolism import BufferSentence, UnclosedEvent
from ..utils.text import (
    segment_sentences,
    format_indexed_text,
    decode_indices,
    normalize_indices,
)

logger = logging.getLogger(__name__)


class CompletedFragment(BaseModel):
    content_raw: str
    continuation_of: Optional[str] = None
    # 本轮缓存中的 1-based 序号。解交织结果用它回写稳定编号，不信模型自填的 continues。
    source_indices: list[int] = Field(default_factory=list)
    # 分裂标记：True 表示该已闭环片段是对某条过长未完成事件的 80% 前缀。
    # ``split_id`` 与同一 BoundaryResult 中某条 NewUnclosed.split_id 配对。
    is_split_prefix: bool = False
    split_id: Optional[str] = None
    # 新协议：indices 已含来源线的句子，原文不再与旧线拼接一次。
    content_includes_source: bool = False


class NewUnclosed(BaseModel):
    content: str
    logical_gaps: Optional[str] = None
    source_indices: list[int] = Field(default_factory=list)
    # 分裂尾段标记：若非 None，则指向同一 ``BoundaryResult`` 中
    # 某条 ``is_split_prefix=True`` 的 ``CompletedFragment``。
    split_id: Optional[str] = None
    continuation_of: Optional[str] = None


class BoundaryResult(BaseModel):
    completed_events: list[CompletedFragment] = Field(default_factory=list)
    new_unclosed: list[NewUnclosed] = Field(default_factory=list)
    no_form_texts: list[str] = Field(default_factory=list)
    pending_texts: list[str] = Field(default_factory=list)
    # True：本轮按 Memory Buffer 解交织。编号由程序维护，未划走的句子退回缓存。
    disentangle: bool = False
    unclaimed_indices: list[int] = Field(default_factory=list)


class BoundaryDetectionSkill:
    """LLM-driven boundary detector producing structured JSON for metabolism.

    输入当前残影文本、本轮用户输入与未完成事件摘要（残影与本轮内容已合并入分句编码表，不重复贴全文），调用 ``boundary_detection`` 模型输出
    ``completed_events`` / ``new_unclosed`` / 可选的分裂配对 ``split_id``，供 ``MetabolismService``
    封存或挂起（白皮书 4.1、1.1.7 防碎片化条款体现在系统/用户 prompt 中）。
    """

    @staticmethod
    def _collect_object_indices(raw: list, n_sent: int) -> set[int]:
        claimed: set[int] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            indices = item.get("indices")
            if not isinstance(indices, list):
                continue
            claimed.update(i for i in normalize_indices(indices) if 1 <= i <= n_sent)
        return claimed

    @staticmethod
    def _collect_legacy_indices(raw: list, n_sent: int) -> set[int]:
        claimed: set[int] = set()

        def _walk(node: object) -> None:
            if isinstance(node, bool):
                return
            if isinstance(node, (int, float)):
                idx = int(node)
                if 1 <= idx <= n_sent:
                    claimed.add(idx)
                return
            if isinstance(node, list):
                for child in node:
                    _walk(child)

        _walk(raw)
        return claimed

    @staticmethod
    def _decode_new_unclosed_list(raw: list, sentences: list[str]) -> list[NewUnclosed]:
        """Map legacy ``new_unclosed_indices`` JSON to ``NewUnclosed`` list.

        - **扁平数字列表** ``[1,2,3]``：视为**同一条**未完成叙事内连续句子（只落库一行）。
        - **嵌套列表** ``[[1,2],[8,9]]``：多线程时多条未完成，每组一行。
        若对扁平表逐项解码，会误将 ``[1,2,3]`` 拆成 3 条只含一句的未完成（DB 里「一条叙事多行」）
        —— 这是此前异常膨胀的主要原因。

        注意：该路径**无法**声明 ``split_id``，即旧格式不支持分裂配对。若需要分裂
        配对，应走 ``_decode_new_unclosed_objects``（新格式 ``new_unclosed: [{{ indices, split_id }}]``）。
        """
        if not raw:
            return []
        is_flat_indices = all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for x in raw
        )
        if is_flat_indices:
            idxs = [int(x) for x in raw]
            content = decode_indices(sentences, idxs)
            if not content:
                return []
            return [NewUnclosed(content=content, logical_gaps=None)]

        new_unc: list[NewUnclosed] = []
        for item in raw:
            if isinstance(item, list):
                content = decode_indices(sentences, item)
            else:
                content = decode_indices(sentences, [int(item)])
            if content:
                new_unc.append(NewUnclosed(content=content, logical_gaps=None))
        return new_unc

    @staticmethod
    def _decode_new_unclosed_objects(
        raw: list,
        sentences: list[str],
    ) -> list[NewUnclosed]:
        """New-format decoder: ``[{ "indices": [...], "split_id": "SP-1"? }, ...]``.

        每个对象独立落成一条 ``NewUnclosed``；其 ``split_id`` 在本批 ``BoundaryResult`` 内
        必须与某条 ``completed_events`` 的 ``split_id`` 配对（配对校验交由 ``BoundaryForceThresholdEvaluator``）。
        """
        if not raw:
            return []
        new_unc: list[NewUnclosed] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            indices = item.get("indices")
            if not isinstance(indices, list):
                continue
            norm_idx: list[int] = normalize_indices(indices)
            if not norm_idx:
                continue
            content = decode_indices(sentences, norm_idx)
            if not content:
                continue
            split_id = item.get("split_id")
            if split_id is not None and not isinstance(split_id, str):
                split_id = None
            logical_gaps = item.get("logical_gaps")
            if logical_gaps is not None and not isinstance(logical_gaps, str):
                logical_gaps = None
            new_unc.append(NewUnclosed(
                content=content,
                logical_gaps=logical_gaps,
                split_id=split_id,
            ))
        return new_unc

    def __init__(self, llm: LLMProvider, config: REMSConfig):
        self._llm = llm
        self._config = config

    def detect(
        self,
        shadow_content: str,
        current_input: str,
        unclosed_events: list[UnclosedEvent] | None = None,
        buffer_items: list[BufferSentence] | None = None,
    ) -> BoundaryResult:
        if buffer_items is not None:
            sentences = [item.text for item in buffer_items if item.text]
        else:
            ctx = self.build_index_context(shadow_content, current_input, unclosed_events)
            sentences = ctx["sentences"]
        ev_len = int(self._config.event_min_chars or 0)
        factor = float(self._config.event_len_factor or 2)
        user_msg = BOUNDARY_USER.format(
            memory_buffer=format_indexed_text(sentences) or "（无）",
            ev_len=ev_len,
            k=f"{factor:g}",
        )
        system_msg = build_user_mode_block(self._config) + BOUNDARY_SYSTEM

        data = self._llm.complete_json(
            "boundary_detection",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return self.parse_response(data, sentences, shadow_count=0, formed_indices=set())

    @staticmethod
    def _line_label(start: int, end: int) -> str:
        if start == end:
            return f"序号 {start}"
        return f"序号 {start}–{end}"

    @staticmethod
    def _append_lines(
        lines: list[UnclosedEvent],
        sentences: list[str],
    ) -> tuple[str, set[int]]:
        labels: list[str] = []
        owned: set[int] = set()
        for ue in lines:
            parts = segment_sentences(ue.merged_content)
            if not parts:
                continue
            start = len(sentences) + 1
            sentences.extend(parts)
            end = len(sentences)
            owned.update(range(start, end + 1))
            labels.append(f"- {ue.id}：{BoundaryDetectionSkill._line_label(start, end)}")
        return ("\n".join(labels) if labels else "（无）"), owned

    @staticmethod
    def build_index_context(
        shadow_content: str,
        current_input: str,
        unclosed_events: list[UnclosedEvent] | None = None,
    ) -> dict:
        """Build sentence index table for boundary (shared by standalone + session Turn3)."""
        lines = list(unclosed_events or [])
        formed = [ue for ue in lines if (ue.formation_role or "residual") == "formed"]
        rejudge = [ue for ue in lines if (ue.formation_role or "residual") == "rejudge"]
        residual = [
            ue for ue in lines
            if (ue.formation_role or "residual") not in ("formed", "rejudge")
        ]
        sentences: list[str] = []
        formed_lines, formed_indices = BoundaryDetectionSkill._append_lines(formed, sentences)
        residual_lines, _residual_idx = BoundaryDetectionSkill._append_lines(residual, sentences)
        _rejudge_lines, _rejudge_idx = BoundaryDetectionSkill._append_lines(rejudge, sentences)
        prior_count = len(sentences)
        if not lines and shadow_content:
            shadow_sentences = segment_sentences(shadow_content)
            sentences.extend(shadow_sentences)
            prior_count = len(sentences)
            if shadow_sentences:
                residual_lines = (
                    f"- ：{BoundaryDetectionSkill._line_label(1, len(shadow_sentences))}"
                )
        current_sentences = segment_sentences(current_input) if current_input else []
        current_start = len(sentences) + 1
        sentences.extend(current_sentences)
        parts: list[str] = []
        cursor = 1
        for label, group in (
            ("已形成尚未封存", formed),
            ("已有残影", residual),
            ("待重判", rejudge),
        ):
            count = 0
            for ue in group:
                count += len(segment_sentences(ue.merged_content))
            if count:
                parts.append(f"【{cursor}..{cursor + count - 1}】={label}")
                cursor += count
        if not lines and shadow_content and prior_count:
            parts.append(f"【1..{prior_count}】=已有残影")
            cursor = prior_count + 1
        if current_sentences:
            parts.append(f"【{current_start}..{len(sentences)}】=本轮新输入")
        range_hint = f"（{'；'.join(parts)}）" if parts else "（无任何输入）"
        return {
            "formed_lines": formed_lines,
            "residual_lines": residual_lines,
            "sentences": sentences,
            "range_hint": range_hint,
            "indexed_input": format_indexed_text(sentences),
            "shadow_count": prior_count,
            "formed_indices": formed_indices,
        }

    @staticmethod
    def _continues_id(raw: object) -> Optional[str]:
        if raw is None or raw == "null":
            return None
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    def _parse_formation(
        self,
        data: dict,
        sentences: list[str],
        formed_indices: set[int],
    ) -> BoundaryResult:
        n_sent = len(sentences)
        claimed: set[int] = set()
        completed: list[CompletedFragment] = []
        for item in data.get("events") or []:
            if not isinstance(item, dict):
                continue
            indices = normalize_indices(item.get("indices") or item.get("content_raw_indices") or [])
            indices = [i for i in indices if 1 <= i <= n_sent]
            if not indices:
                continue
            claimed.update(indices)
            content = decode_indices(sentences, indices)
            if not content:
                continue
            completed.append(CompletedFragment(
                content_raw=content,
                continuation_of=None,
                content_includes_source=True,
                source_indices=list(indices),
            ))

        new_unc: list[NewUnclosed] = []
        for item in data.get("residual") or []:
            if not isinstance(item, dict):
                continue
            indices = normalize_indices(item.get("indices") or [])
            indices = [i for i in indices if 1 <= i <= n_sent and i not in claimed]
            if not indices:
                continue
            claimed.update(indices)
            content = decode_indices(sentences, indices)
            if not content:
                continue
            new_unc.append(NewUnclosed(
                content=content,
                continuation_of=None,
                source_indices=list(indices),
            ))

        no_form_idx = [
            i for i in normalize_indices(data.get("no_form") or [])
            if 1 <= i <= n_sent and i not in claimed
        ]
        missing = [
            i for i in range(1, n_sent + 1)
            if i not in claimed and i not in formed_indices and i not in no_form_idx
        ]
        unclaimed = list(dict.fromkeys([*no_form_idx, *missing]))
        return BoundaryResult(
            completed_events=completed,
            new_unclosed=new_unc,
            disentangle=True,
            unclaimed_indices=unclaimed,
        )

    def parse_response(
        self,
        data: dict,
        sentences: list[str],
        shadow_count: int = 0,
        formed_indices: set[int] | None = None,
    ) -> BoundaryResult:
        if any(key in data for key in ("events", "residual", "no_form")):
            return self._parse_formation(data, sentences, formed_indices or set())
        completed: list[CompletedFragment] = []
        claimed: set[int] = set()
        n_sent = len(sentences)
        for item in data.get("completed_events", []):
            if not isinstance(item, dict):
                continue
            indices = normalize_indices(item.get("content_raw_indices", []))
            claimed.update(i for i in indices if 1 <= i <= n_sent)
            continuation_of = item.get("continuation_of")
            # 续写去重：命中 continuation_of 时，旧残影内容由后端合并 UC 自动补上；
            # 这里剥掉落在旧残影区间内的句子索引，避免 ue.merged_content + 片段重复。
            # 这些旧序号已经计入 claimed，不会再被收成另一条未完成线。
            if continuation_of and shadow_count > 0:
                indices = [i for i in indices if i > shadow_count]
            extracted_raw = decode_indices(sentences, indices)
            if not extracted_raw and not continuation_of:
                continue

            split_id = item.get("split_id")
            if split_id is not None and not isinstance(split_id, str):
                split_id = None
            is_split_prefix = bool(item.get("is_split_prefix", False))
            if split_id and not is_split_prefix:
                is_split_prefix = True
            if is_split_prefix and not split_id:
                is_split_prefix = False

            completed.append(CompletedFragment(
                content_raw=extracted_raw,
                continuation_of=continuation_of,
                is_split_prefix=is_split_prefix,
                split_id=split_id,
            ))

        new_unc: list[NewUnclosed] = []
        new_obj = data.get("new_unclosed")
        if isinstance(new_obj, list) and new_obj and all(isinstance(x, dict) for x in new_obj):
            claimed.update(self._collect_object_indices(new_obj, n_sent))
            new_unc = self._decode_new_unclosed_objects(new_obj, sentences)
        else:
            legacy_raw = data.get("new_unclosed_indices") or data.get("new_unclosed") or []
            if isinstance(legacy_raw, list):
                claimed.update(self._collect_legacy_indices(legacy_raw, n_sent))
                new_unc = self._decode_new_unclosed_list(legacy_raw, sentences)

        missing = [i for i in range(1, n_sent + 1) if i not in claimed]
        if missing:
            leftover = decode_indices(sentences, missing)
            if leftover:
                new_unc.append(NewUnclosed(content=leftover, logical_gaps=None))

        return BoundaryResult(
            completed_events=completed,
            new_unclosed=new_unc,
        )
