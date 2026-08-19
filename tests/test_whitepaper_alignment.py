"""Whitepaper-alignment unit tests for boundary / role / event-service interfaces.

升级历史：
    - boundary 输出字段从旧的 ``remaining_shadow`` 改为 ``new_unclosed_indices``
      （白皮书 §4.1：残影 = 未完成事件拼接，不再独立持久化）。
    - 摘要预算从 boundary prompt 移到 ``EventEnrichmentSkill``，本测试验证 boundary
      prompt 显式带"残影 / 当前输入"区间标记（P1-9）。
    - ``EventService.seal_event`` 签名变更：摘要由 enrichment 单次调用产出，
      不再接受 ``pre_summaries``；构造签名换成 ``EventEnrichmentSkill`` + ``RoleService``。
"""

from __future__ import annotations

import pytest

from rems.models.event import CompressionBudget
from rems.services.event_service import EventService
from rems.services.role_service import RoleService
from rems.skills.boundary_detection import BoundaryDetectionSkill
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.repository import EventRepository, RoleRepository


class _DummyJSONLLM:
    """JSON-only LLM stub that records the most recent message list."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.last_messages: list[dict[str, str]] | None = None

    def complete_json(
        self,
        task_type: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
    ) -> dict:
        self.last_messages = messages
        return self.payload

    def complete(self, task_type: str, messages, *, temperature=None) -> str:
        self.last_messages = messages
        return ""


class _NoopVectorStore:
    def add_event(self, event_id: str, text: str, metadata: dict) -> None:
        return None

    def add_white_painting(self, role_id, event_id, text, metadata=None):
        return None


def test_boundary_detect_returns_new_unclosed_only(config):
    llm = _DummyJSONLLM(
        {
            "completed_events": [
                {
                    "content_raw_indices": [1],
                    "continuation_of": None,
                }
            ],
            # 第二句应被识别为新未完成（残影由未完成库重建，不再有独立 remaining_shadow 字段）。
            "new_unclosed_indices": [2],
        }
    )
    skill = BoundaryDetectionSkill(llm, config)

    result = skill.detect("", "甲。乙。")

    assert result.completed_events[0].content_raw == "甲。"
    assert len(result.new_unclosed) == 1
    assert "乙" in result.new_unclosed[0].content
    assert not hasattr(result, "remaining_shadow")


def test_boundary_prompt_marks_shadow_and_current_ranges(config):
    """P1-9：prompt 必须显式标注残影 / 当前输入的句子区间，否则模型无法分辨边界。"""
    llm = _DummyJSONLLM(
        {
            "completed_events": [],
            "new_unclosed_indices": [],
        }
    )
    skill = BoundaryDetectionSkill(llm, config)

    skill.detect("旧残影第一句。", "新输入第一句。新输入第二句。")
    user_prompt = llm.last_messages[1]["content"]

    assert "既有残影" in user_prompt
    assert "本轮新输入" in user_prompt


def test_role_extraction_prompt_includes_snapshot_budget(config):
    llm = _DummyJSONLLM({"roles": []})
    skill = RoleExtractionSkill(llm, config)
    budget = CompressionBudget(
        raw_len=120,
        total_budget=36,
        summary_level_budgets={},
        snapshot_level_budgets={"L3": 24, "L2": 14, "L1": 9},
        wp_budget_per_role=7,
        decoration_budget=3,
    )

    skill.extract("张三和李四讨论项目进展。", budget=budget)
    user_prompt = llm.last_messages[1]["content"]

    assert "角色快照预算表" in user_prompt
    assert "L3: 24 字以内" in user_prompt


def test_event_service_seal_event_basic(config, db, fake_llm):
    """新签名：EventService(config, llm, event_repo, vector_store, enrichment_skill, role_service=...)。"""
    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store = _NoopVectorStore()
    role_skill = RoleExtractionSkill(fake_llm, config)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config, role_fallback=role_skill)
    role_service = RoleService(config, role_repo, role_skill, llm=fake_llm, vector_store=vector_store)

    service = EventService(
        config, fake_llm, event_repo, vector_store,
        enrichment_skill, role_service=role_service,
    )

    # enrichment 单次调用同时返回摘要 + 角色（白皮书 1.1.3、2.1）。
    fake_llm.push_response({
        "summaries": {"L1": "张三和李四讨论项目"},
        "roles": [
            {
                "role_id": None,
                "name": "张三",
                "importance": "S",
                "snapshot": {"l1_mention": "张三", "l2_interaction": "提议", "l3_decision": "推动"},
                "emotion": {"joy": 0.5, "trust": 0.6},
            }
        ],
    })

    event = service.seal_event("张三和李四讨论项目进展。")
    assert event.summaries.get("L1") == "张三和李四讨论项目"
    assert len(event.role_list) >= 1
