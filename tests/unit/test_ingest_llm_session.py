"""Tests for IngestLlmSession multi-turn ingest LLM flow."""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.llm.ingest_session import IngestLlmSession
from rems.llm.prompts import INGEST_CONTENT_RAW_REFERENCE, ROLE_EXTRACTION_USER
from rems.skills.boundary_detection import BoundaryDetectionSkill


@pytest.fixture
def session(fake_llm, config):
    config.recall_query_compress_enabled = True
    config.recall_query_compress_min_chars = 10
    config.ingest_llm_session_enabled = True
    boundary = BoundaryDetectionSkill(fake_llm, config)
    return IngestLlmSession(fake_llm, config, boundary)


class TestIngestLlmSession:
    def test_three_turns_shared_context_user(self, session, fake_llm):
        full_text = "张三去了北京。李四留在上海等待消息。"
        fake_llm.push_response({
            "act_query": "张三赴北京办事，李四留在上海等待消息，王五在途中传递口信，众人各自分头行动。",
        })
        fake_llm.push_response({"roles": [{"name": "张三", "importance": "B", "snapshot": {"l1_mention": "张三"}}]})
        fake_llm.push_response({
            "completed_events": [{"content_raw_indices": [1]}],
            "new_unclosed": [{"indices": [2]}],
        })

        act = session.compress_search_text(full_text)
        assert "张三" in act
        assert session.turn == 1

        roles = session.extract_roles_followup()
        assert len(roles.roles) == 1
        assert session.turn == 2

        result = session.detect_boundary_followup("", "李四留在上海。", None)
        assert len(result.completed_events) == 1
        assert session.turn == 3

    def test_role_turn_uses_original_user_template(self, session, fake_llm, config):
        full_text = "测试原文内容足够长以触发压缩路径" * 5
        fake_llm.push_response({
            "act_query": (
                "测试原文内容足够长以触发压缩路径，其中包含多个人物与动作链，"
                "用于验证检索摘要不应被压成一句标题，需保留足够细节供向量匹配。"
                "凤姐设圈套约贾瑞，贾蔷贾蓉捉奸逼写借据，贾母王夫人各执一词。"
            ),
        })
        fake_llm.push_response({"roles": []})

        session.compress_search_text(full_text)
        session.extract_roles_followup()

        last_call = fake_llm.calls[-1]["messages"]
        task_user = last_call[-1]["content"]
        expected = ROLE_EXTRACTION_USER.format(
            known_roles="无已知角色",
            content_raw=INGEST_CONTENT_RAW_REFERENCE,
            snapshot_budgets="按系统默认要求",
        )
        assert task_user == expected
        assert full_text not in task_user
        assert "你是 REMS 角色提取组件" in last_call[0]["content"]

    def test_turn2_requires_turn1(self, session):
        with pytest.raises(RuntimeError, match="Turn2"):
            session.extract_roles_followup()
