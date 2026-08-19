"""Tests for pre-recall focus role heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field

from rems.ingest.focus_roles import match_focus_role_ids
from rems.models.role import Role


@dataclass
class _FakeRoleRepo:
    roles: list[Role] = field(default_factory=list)

    def list_all(self) -> list[Role]:
        return self.roles


class TestMatchFocusRoleIds:
    def test_traditional_text_matches_simplified_registry(self):
        repo = _FakeRoleRepo(
            roles=[
                Role(role_id="ROL-baoyu", name="宝玉"),
                Role(role_id="ROL-feng", name="凤姐"),
                Role(role_id="ROL-qin", name="秦钟"),
                Role(role_id="ROL-cm", name="彩明"),
            ]
        )
        raw = "寶玉拉了秦鐘，直至抱廈．鳳姐才吃飯"
        focus = match_focus_role_ids(repo, raw_input=raw)
        assert focus == {"ROL-baoyu", "ROL-feng", "ROL-qin"}

    def test_act_query_supplements_courtesy_names(self):
        repo = _FakeRoleRepo(
            roles=[
                Role(role_id="ROL-dy", name="黛玉"),
                Role(role_id="ROL-baoyu", name="宝玉"),
            ]
        )
        raw = "鳳姐向寶玉笑道：你林妹妹可在咱們家住長了。"
        act = "凤姐对宝玉说林妹妹可长住，宝玉担心黛玉哭泣。"
        focus = match_focus_role_ids(repo, raw_input=raw, extra_texts=[act])
        assert "ROL-dy" in focus
        assert "ROL-baoyu" in focus

    def test_alias_on_role(self):
        repo = _FakeRoleRepo(
            roles=[
                Role(role_id="ROL-dy", name="黛玉", aliases=["林姑娘"]),
            ]
        )
        focus = match_focus_role_ids(repo, raw_input="林姑娘哭了")
        assert focus == {"ROL-dy"}

    def test_narrative_trigger_lin_gu_lao_ye(self):
        repo = _FakeRoleRepo(roles=[Role(role_id="ROL-lrh", name="林如海")])
        raw = "昭儿道：林姑老爷是九月初三日巳時沒的。"
        focus = match_focus_role_ids(repo, raw_input=raw)
        assert focus == {"ROL-lrh"}
