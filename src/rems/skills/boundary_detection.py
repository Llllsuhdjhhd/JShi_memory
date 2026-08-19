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
from ..models.metabolism import UnclosedEvent
from ..utils.text import segment_sentences, format_indexed_text, decode_indices

logger = logging.getLogger(__name__)


class CompletedFragment(BaseModel):
    content_raw: str
    continuation_of: Optional[str] = None
    # 分裂标记：True 表示该已闭环片段是对某条过长未完成事件的 80% 前缀。
    # ``split_id`` 与同一 BoundaryResult 中某条 NewUnclosed.split_id 配对。
    is_split_prefix: bool = False
    split_id: Optional[str] = None


class NewUnclosed(BaseModel):
    content: str
    logical_gaps: Optional[str] = None
    # 分裂尾段标记：若非 None，则指向同一 ``BoundaryResult`` 中
    # 某条 ``is_split_prefix=True`` 的 ``CompletedFragment``。
    split_id: Optional[str] = None


class BoundaryResult(BaseModel):
    completed_events: list[CompletedFragment] = Field(default_factory=list)
    new_unclosed: list[NewUnclosed] = Field(default_factory=list)


class BoundaryDetectionSkill:
    """LLM-driven boundary detector producing structured JSON for metabolism.

    输入当前残影文本、本轮用户输入与未完成事件摘要（残影与本轮内容已合并入分句编码表，不重复贴全文），调用 ``boundary_detection`` 模型输出
    ``completed_events`` / ``new_unclosed`` / 可选的分裂配对 ``split_id``，供 ``MetabolismService``
    封存或挂起（白皮书 4.1、1.1.7 防碎片化条款体现在系统/用户 prompt 中）。
    """

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
            norm_idx: list[int] = []
            for v in indices:
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    norm_idx.append(int(v))
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
    ) -> BoundaryResult:
        ctx = self.build_index_context(shadow_content, current_input, unclosed_events)
        force_threshold = int(self._config.len_msg * self._config.unclosed_force_ratio)
        user_msg = BOUNDARY_USER.format(
            unclosed_summary=ctx["unclosed_summary"],
            indexed_input=ctx["indexed_input"],
            range_hint=ctx["range_hint"],
            force_threshold=force_threshold,
        )
        system_msg = build_user_mode_block(self._config) + BOUNDARY_SYSTEM

        data = self._llm.complete_json(
            "boundary_detection",
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        return self.parse_response(data, ctx["sentences"])

    @staticmethod
    def build_index_context(
        shadow_content: str,
        current_input: str,
        unclosed_events: list[UnclosedEvent] | None = None,
    ) -> dict:
        """Build sentence index table for boundary (shared by standalone + session Turn3)."""
        unclosed_summary = "无" if not unclosed_events else "\n".join(
            f"- ID={ue.id}, 片段={ue.merged_content[:80]}…, 缺={ue.logical_gaps or '未知'}"
            for ue in (unclosed_events or [])
        )
        shadow_sentences = segment_sentences(shadow_content) if shadow_content else []
        current_sentences = segment_sentences(current_input) if current_input else []
        sentences = shadow_sentences + current_sentences
        shadow_count = len(shadow_sentences)
        if shadow_count > 0 and current_sentences:
            range_hint = (
                f"（【1..{shadow_count}】=既有残影；"
                f"【{shadow_count + 1}..{len(sentences)}】=本轮新输入）"
            )
        elif shadow_count > 0:
            range_hint = f"（【1..{shadow_count}】=既有残影；本轮无新输入）"
        elif current_sentences:
            range_hint = f"（【1..{len(sentences)}】=本轮新输入；无既有残影）"
        else:
            range_hint = "（无任何输入）"
        return {
            "unclosed_summary": unclosed_summary,
            "sentences": sentences,
            "range_hint": range_hint,
            "indexed_input": format_indexed_text(sentences),
            "shadow_count": shadow_count,
        }

    def parse_response(self, data: dict, sentences: list[str]) -> BoundaryResult:
        completed: list[CompletedFragment] = []
        for item in data.get("completed_events", []):
            if not isinstance(item, dict):
                continue
            indices = item.get("content_raw_indices", [])
            extracted_raw = decode_indices(sentences, indices)
            if not extracted_raw:
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
                continuation_of=item.get("continuation_of"),
                is_split_prefix=is_split_prefix,
                split_id=split_id,
            ))

        new_unc: list[NewUnclosed] = []
        new_obj = data.get("new_unclosed")
        if isinstance(new_obj, list) and new_obj and all(isinstance(x, dict) for x in new_obj):
            new_unc = self._decode_new_unclosed_objects(new_obj, sentences)
        else:
            legacy_raw = data.get("new_unclosed_indices") or data.get("new_unclosed") or []
            if isinstance(legacy_raw, list):
                new_unc = self._decode_new_unclosed_list(legacy_raw, sentences)

        return BoundaryResult(
            completed_events=completed,
            new_unclosed=new_unc,
        )
