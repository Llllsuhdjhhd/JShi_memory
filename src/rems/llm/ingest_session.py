from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from ..config import REMSConfig
from ..embedding.act_text import clamp_embed_text
from ..llm.prompts import (
    BOUNDARY_SYSTEM,
    BOUNDARY_USER,
    INGEST_CONTENT_RAW_REFERENCE,
    INGEST_CONTEXT_USER,
    RECALL_QUERY_COMPRESS_SYSTEM,
    ROLE_EXTRACTION_SYSTEM,
    ROLE_EXTRACTION_USER,
    build_user_mode_block,
)
from ..models.metabolism import UnclosedEvent
from ..models.role import Role
from ..skills.boundary_detection import BoundaryDetectionSkill, BoundaryResult
from ..skills.recall_query_compress import compress_act_query_from_llm, format_compress_followup
from ..skills.role_extraction import RoleExtractionResult, parse_role_extraction_data

if TYPE_CHECKING:
    from ..llm.provider import LLMProvider
    from ..skills.boundary_detection import BoundaryDetectionSkill as BoundarySkillType

logger = logging.getLogger(__name__)


class IngestLlmSession:
    """Single-ingest LLM session: one shared context user + per-task system/user.

    每次 API 调用结构：``system（任务专用） + 首条 context user（全文仅此处） + task user``。
    角色/边界 user 模板与独立 Skill 路径一致（``ROLE_EXTRACTION_USER`` / ``BOUNDARY_USER``）。
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: REMSConfig,
        boundary_skill: BoundarySkillType,
    ):
        self._llm = llm
        self._config = config
        self._boundary = boundary_skill
        self._lock = threading.Lock()
        self._context_message: dict[str, str] | None = None
        self._search_text: str = ""
        self._turn: int = 0

    def reset(self) -> None:
        with self._lock:
            self._context_message = None
            self._search_text = ""
            self._turn = 0

    @property
    def turn(self) -> int:
        return self._turn

    def _session_model(self) -> str | None:
        return self._config.ingest_llm_session_task_model

    def _task_messages(self, system_content: str, task_user: str) -> list[dict[str, str]]:
        if self._context_message is None:
            raise RuntimeError("ingest session context not initialized")
        return [
            {"role": "system", "content": system_content},
            self._context_message,
            {"role": "user", "content": task_user},
        ]

    def compress_search_text(self, text: str) -> str:
        """Turn1: shared context user + compress task; blocks recall until done."""
        text = (text or "").strip()
        if not text:
            return ""

        cap = self._config.tri_band.query_act_compress_max_chars
        skip_min = self._config.recall_query_compress_min_chars

        with self._lock:
            self._context_message = {
                "role": "user",
                "content": INGEST_CONTEXT_USER.format(content=text),
            }
            self._search_text = text

        skip_llm = (
            not self._config.recall_query_compress_enabled
            or (len(text) <= skip_min and len(text) <= cap)
        )
        if skip_llm:
            act_query = clamp_embed_text(text, cap)
            with self._lock:
                self._turn = 1
            return act_query

        task_user = format_compress_followup(self._config, len(text))
        messages = self._task_messages(RECALL_QUERY_COMPRESS_SYSTEM, task_user)
        act_query = compress_act_query_from_llm(
            self._llm,
            self._config,
            messages,
            len(text),
            model=self._session_model(),
        )
        with self._lock:
            self._turn = 1
        return act_query

    def extract_roles_followup(
        self,
        known_roles: list[Role] | None = None,
        *,
        snapshot_budgets_text: str = "按系统默认要求",
    ) -> RoleExtractionResult:
        """Turn2: original ``ROLE_EXTRACTION_*`` prompts; no full-text re-paste."""
        with self._lock:
            if self._turn < 1 or self._context_message is None:
                raise RuntimeError("ingest session Turn2 requires Turn1 compress_search_text")
            known_desc = "无已知角色" if not known_roles else "\n".join(
                f"- {r.role_id}: {r.name} ({r.entity_type}), 别名={r.aliases}"
                for r in (known_roles or [])
            )
            task_user = ROLE_EXTRACTION_USER.format(
                known_roles=known_desc,
                content_raw=INGEST_CONTENT_RAW_REFERENCE,
                snapshot_budgets=snapshot_budgets_text,
            )
            system_msg = build_user_mode_block(self._config) + ROLE_EXTRACTION_SYSTEM
            messages = self._task_messages(system_msg, task_user)
            data = self._llm.complete_json(
                "role_extraction",
                messages,
                temperature=0.1,
                model=self._session_model(),
            )
            self._turn = 2
            return parse_role_extraction_data(data)

    def detect_boundary_followup(
        self,
        shadow_content: str,
        current_input: str,
        unclosed_events: list[UnclosedEvent] | None = None,
    ) -> BoundaryResult:
        """Turn3: original ``BOUNDARY_*`` prompts; indexed table only (no raw re-paste)."""
        ctx = BoundaryDetectionSkill.build_index_context(
            shadow_content, current_input, unclosed_events,
        )
        force_threshold = int(self._config.len_msg * self._config.unclosed_force_ratio)
        task_user = BOUNDARY_USER.format(
            unclosed_summary=ctx["unclosed_summary"],
            indexed_input=ctx["indexed_input"],
            range_hint=ctx["range_hint"],
            force_threshold=force_threshold,
        )
        system_msg = build_user_mode_block(self._config) + BOUNDARY_SYSTEM

        with self._lock:
            if self._turn < 1 or self._context_message is None:
                raise RuntimeError("ingest session Turn3 requires Turn1")
            messages = self._task_messages(system_msg, task_user)
            data = self._llm.complete_json(
                "boundary_detection",
                messages,
                temperature=0.1,
                model=self._session_model(),
            )
            self._turn = 3
            return self._boundary.parse_response(data, ctx["sentences"])

    def audit_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "turn": self._turn,
                "search_text_chars": len(self._search_text),
                "context_user_chars": len((self._context_message or {}).get("content") or ""),
            }
