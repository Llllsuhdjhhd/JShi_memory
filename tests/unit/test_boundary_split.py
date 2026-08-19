"""Tests for the 80/20 forced-split pipeline.

覆盖面：
    - ``BoundaryForceThresholdEvaluator`` 的三条规则（oversized_uc / broken_split_pair / split_ratio_off）。
    - ``OverlongUCSplitSkill`` 对单条过长文本的 LLM 驱动分裂 + 字符等价校验。
    - ``OverlongUCSplitRemediator`` 把评估报告里的 oversized_uc 转成 prefix_completed + tail_unclosed；
      并在 LLM 放弃切分时把原条保留（不崩）。
    - ``MetabolismService`` 端到端：
        * 前缀被 seal_event 持久化，tail UC 带 ``split_prefix_event_ids``；
        * tail UC 下一轮闭环时，新事件继承前缀链，并反向把自己追加到前缀事件的 successor；
        * oversized UC（模型 + 修复都失败）不走 force-seal，而是落库为 oversized=True。
    - ``RecallService._expand_split_prefixes`` 命中 tail 事件时把前缀事件硬性拉进回忆块。
"""

from __future__ import annotations

import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.models.metabolism import UnclosedEvent
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.services.recall_service import RecallService
from rems.skills.boundary_detection import (
    BoundaryDetectionSkill,
    BoundaryResult,
    CompletedFragment,
    NewUnclosed,
)
from rems.skills.boundary_split import (
    BoundaryForceThresholdEvaluator,
    BoundarySkillContext,
    OverlongUCSplitRemediator,
    OverlongUCSplitSkill,
)
from rems.skills.event_enrichment import EventEnrichmentSkill
from rems.skills.role_extraction import RoleExtractionSkill
from rems.storage.database import Database
from rems.embedding.tri_band import TriBandEncoder
from rems.storage.repository import EventRepository, MetabolismRepository, RoleRepository

from ..conftest import FakeLLM


# =====================================================================
# BoundaryForceThresholdEvaluator
# =====================================================================

class TestBoundaryForceThresholdEvaluator:
    def _ctx(self, threshold: int = 100) -> BoundarySkillContext:
        return BoundarySkillContext(
            force_threshold=threshold,
            split_ratio_target=0.8,
            split_ratio_min=0.7,
            split_ratio_max=0.9,
        )

    def test_ok_when_no_oversized_uc(self, config):
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[CompletedFragment(content_raw="short closed")],
            new_unclosed=[NewUnclosed(content="short open")],
        )
        report = ev.evaluate(self._ctx(100), result)
        assert report.ok

    def test_flags_oversized_uc(self, config):
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            new_unclosed=[NewUnclosed(content="x" * 500)],
        )
        report = ev.evaluate(self._ctx(100), result)
        assert not report.ok
        assert report.has_code("oversized_uc")

    def test_paired_split_not_flagged_as_oversized(self, config):
        """合法的 80/20 分裂：前缀 200 字 + 尾部 50 字（总 250 > 100 但已配对）。"""
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[CompletedFragment(
                content_raw="p" * 200, is_split_prefix=True, split_id="SP-1",
            )],
            new_unclosed=[NewUnclosed(content="t" * 50, split_id="SP-1")],
        )
        report = ev.evaluate(self._ctx(100), result)
        # 尾部 50 < 100，不越阈值；配对完整；ok。
        assert report.ok

    def test_tail_still_oversized_is_error(self, config):
        """配对了但尾部本身仍过长：应报 split_tail_still_oversized。"""
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[CompletedFragment(
                content_raw="p" * 100, is_split_prefix=True, split_id="SP-1",
            )],
            new_unclosed=[NewUnclosed(content="t" * 500, split_id="SP-1")],
        )
        report = ev.evaluate(self._ctx(100), result)
        assert not report.ok
        assert report.has_code("split_tail_still_oversized")

    def test_broken_pair_prefix_without_tail(self, config):
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[CompletedFragment(
                content_raw="p" * 20, is_split_prefix=True, split_id="SP-1",
            )],
            new_unclosed=[],
        )
        report = ev.evaluate(self._ctx(100), result)
        assert not report.ok
        assert report.has_code("broken_split_pair")

    def test_broken_pair_tail_without_prefix(self, config):
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[],
            new_unclosed=[NewUnclosed(content="t" * 10, split_id="SP-orphan")],
        )
        report = ev.evaluate(self._ctx(100), result)
        assert not report.ok
        assert report.has_code("broken_split_pair")

    def test_split_ratio_off_warn_only(self, config):
        """比例 50% 远离 80% 目标：只是 warn，不应让 ok 转 False。"""
        ev = BoundaryForceThresholdEvaluator(config)
        result = BoundaryResult(
            completed_events=[CompletedFragment(
                content_raw="p" * 50, is_split_prefix=True, split_id="SP-1",
            )],
            new_unclosed=[NewUnclosed(content="t" * 50, split_id="SP-1")],
        )
        report = ev.evaluate(self._ctx(200), result)
        assert report.has_code("split_ratio_off")
        assert report.ok  # warn 不阻塞


# =====================================================================
# OverlongUCSplitSkill + Remediator
# =====================================================================

class TestOverlongUCSplitSkill:
    def test_split_happy_path(self, config, fake_llm):
        long_text = "a" * 80 + "b" * 20  # 100 chars
        prefix = "a" * 80
        tail = "b" * 20
        fake_llm.push_response({"prefix_text": prefix, "tail_text": tail})

        skill = OverlongUCSplitSkill(fake_llm, config)
        out = skill.split(long_text)
        assert not out.aborted
        assert out.prefix_text == prefix
        assert out.tail_text == tail

    def test_split_abort_when_llm_says_abort(self, config, fake_llm):
        fake_llm.push_response({"abort": True})
        skill = OverlongUCSplitSkill(fake_llm, config)
        out = skill.split("a" * 50 + "b" * 50)
        assert out.aborted

    def test_split_abort_on_content_mismatch(self, config, fake_llm):
        # LLM 改写了原文 → 校验失败，触发 abort 而非污染结果。
        fake_llm.push_response({
            "prefix_text": "completely unrelated prefix",
            "tail_text": "likewise unrelated tail",
        })
        skill = OverlongUCSplitSkill(fake_llm, config)
        out = skill.split("a" * 80 + "b" * 20)
        assert out.aborted

    def test_split_abort_on_extreme_ratio(self, config, fake_llm):
        # 99% 前缀、1% 尾部：远出区间 → 放弃。
        prefix = "a" * 99
        tail = "b" * 1
        fake_llm.push_response({"prefix_text": prefix, "tail_text": tail})
        skill = OverlongUCSplitSkill(fake_llm, config)
        out = skill.split(prefix + tail)
        assert out.aborted


class TestOverlongUCSplitRemediator:
    def _evaluate(self, config, result, threshold=100):
        ctx = BoundarySkillContext(
            force_threshold=threshold,
            split_ratio_target=0.8,
            split_ratio_min=0.7,
            split_ratio_max=0.9,
        )
        ev = BoundaryForceThresholdEvaluator(config)
        return ctx, ev.evaluate(ctx, result)

    def test_oversized_uc_gets_split_into_pair(self, config, fake_llm):
        oversize_text = "a" * 80 + "b" * 20
        fake_llm.push_response({"prefix_text": "a" * 80, "tail_text": "b" * 20})

        result = BoundaryResult(
            completed_events=[],
            new_unclosed=[NewUnclosed(content=oversize_text)],
        )
        ctx, report = self._evaluate(config, result, threshold=50)
        assert not report.ok

        split_skill = OverlongUCSplitSkill(fake_llm, config)
        remediator = OverlongUCSplitRemediator(split_skill)
        repaired = remediator.remediate(ctx, result, report)
        assert repaired is not None
        assert len(repaired.completed_events) == 1
        assert repaired.completed_events[0].is_split_prefix
        assert repaired.completed_events[0].split_id is not None
        assert len(repaired.new_unclosed) == 1
        assert repaired.new_unclosed[0].split_id == repaired.completed_events[0].split_id

    def test_abort_keeps_original_oversized_uc(self, config, fake_llm):
        fake_llm.push_response({"abort": True})
        oversize_text = "x" * 200
        result = BoundaryResult(
            completed_events=[],
            new_unclosed=[NewUnclosed(content=oversize_text)],
        )
        ctx, report = self._evaluate(config, result, threshold=50)

        split_skill = OverlongUCSplitSkill(fake_llm, config)
        remediator = OverlongUCSplitRemediator(split_skill)
        repaired = remediator.remediate(ctx, result, report)
        # 仍保留原条、无新前缀、保持 oversized（交由 metabolism_service 标 oversized=True）。
        assert repaired is not None
        assert len(repaired.completed_events) == 0
        assert len(repaired.new_unclosed) == 1
        assert repaired.new_unclosed[0].content == oversize_text
        assert repaired.new_unclosed[0].split_id is None

    def test_orphan_prefix_demoted_to_plain_completed(self, config, fake_llm):
        result = BoundaryResult(
            completed_events=[CompletedFragment(
                content_raw="orphan prefix", is_split_prefix=True, split_id="SP-bad",
            )],
            new_unclosed=[],
        )
        ctx, report = self._evaluate(config, result)

        split_skill = OverlongUCSplitSkill(fake_llm, config)
        remediator = OverlongUCSplitRemediator(split_skill)
        repaired = remediator.remediate(ctx, result, report)
        assert repaired is not None
        assert len(repaired.completed_events) == 1
        assert not repaired.completed_events[0].is_split_prefix
        assert repaired.completed_events[0].split_id is None


# =====================================================================
# MetabolismService end-to-end with split
# =====================================================================

@pytest.fixture()
def metabolism_service_with_repair(config: REMSConfig, db: Database, fake_llm: FakeLLM, vector_store):
    event_repo = EventRepository(db)
    meta_repo = MetabolismRepository(db)
    role_skill = RoleExtractionSkill(fake_llm, config)
    enrichment_skill = EventEnrichmentSkill(fake_llm, config, role_fallback=role_skill)
    event_service = EventService(config, fake_llm, event_repo, vector_store, enrichment_skill)
    boundary_skill = BoundaryDetectionSkill(fake_llm, config)
    svc = MetabolismService.with_default_boundary_repair(
        config, meta_repo, boundary_skill, event_service,
        event_repo=event_repo, llm=fake_llm,
    )
    return svc, meta_repo, event_repo


class TestMetabolismServiceSplitFlow:
    def test_boundary_returns_split_pair_is_persisted(
        self, metabolism_service_with_repair, fake_llm
    ):
        """模型自己就给出了 80/20 分裂对：prefix 作为 completed 封存，tail 挂 UC 并带前缀 id。"""
        svc, meta_repo, event_repo = metabolism_service_with_repair
        # 1 句子 = 一整段，防止分句把它拆碎；我们要做"整段都是一条 UC"的效果。
        prefix_text = "p" * 80 + "。"
        tail_text = "t" * 20 + "。"
        full_text = prefix_text + tail_text

        # boundary_detection：返回新格式，prefix + tail 配对。
        fake_llm.push_response({
            "completed_events": [{
                "content_raw_indices": [1],
                "is_split_prefix": True,
                "split_id": "SP-a",
            }],
            "new_unclosed": [{"indices": [2], "split_id": "SP-a"}],
        })
        # enrichment for prefix event
        fake_llm.push_response({"summaries": {"L1": "prefix closed"}, "roles": []})

        events = svc.process_input(full_text)
        assert len(events) == 1
        prefix_event = events[0]

        # UC 应当带上 split_prefix_event_ids=[prefix_event.event_id]
        ues = meta_repo.get_unclosed_events()
        assert len(ues) == 1
        assert ues[0].split_prefix_event_ids == [prefix_event.event_id]
        assert not ues[0].oversized

    def test_tail_close_inherits_prefix_chain_and_appends_successor(
        self, metabolism_service_with_repair, fake_llm
    ):
        """Tail UC 闭环后：新事件继承 split_prefix_event_ids；前缀事件的 split_successor_event_ids 反向追加。"""
        svc, meta_repo, event_repo = metabolism_service_with_repair

        # Round 1：模型自分裂。
        prefix_text = "p" * 80 + "。"
        tail_text = "t" * 20 + "。"
        fake_llm.push_response({
            "completed_events": [{
                "content_raw_indices": [1],
                "is_split_prefix": True,
                "split_id": "SP-a",
            }],
            "new_unclosed": [{"indices": [2], "split_id": "SP-a"}],
        })
        fake_llm.push_response({"summaries": {"L1": "prefix"}, "roles": []})
        svc.process_input(prefix_text + tail_text)
        ues_after_r1 = meta_repo.get_unclosed_events()
        assert len(ues_after_r1) == 1
        prefix_event_id = ues_after_r1[0].split_prefix_event_ids[0]

        # Round 2：新输入 1 句子 → boundary 判定 continuation_of=tail_uc，将 tail+new 闭环。
        tail_uc_id = ues_after_r1[0].id
        fake_llm.push_response({
            "completed_events": [{
                "content_raw_indices": [1, 2],  # shadow (tail) + current input
                "continuation_of": tail_uc_id,
            }],
            "new_unclosed": [],
        })
        fake_llm.push_response({"summaries": {"L1": "closure"}, "roles": []})
        sealed = svc.process_input("收尾句子。")
        assert len(sealed) == 1
        closure_event = sealed[0]

        # 继承 & 反向链都应写入。
        assert prefix_event_id in closure_event.split_prefix_event_ids
        prefix_event = event_repo.get(prefix_event_id)
        assert prefix_event is not None
        assert closure_event.event_id in prefix_event.split_successor_event_ids

    def test_oversized_uc_without_split_falls_back_to_remediation(
        self, metabolism_service_with_repair, fake_llm
    ):
        """模型没自己切 → 规则评估器报 oversized_uc → 走 OverlongUCSplitSkill 补切。"""
        svc, meta_repo, event_repo = metabolism_service_with_repair
        # 用足够长的字符串强制越阈值（1 sentence, 1 segment）。
        force_threshold = int(svc._config.len_msg * svc._config.unclosed_force_ratio)  # noqa: SLF001
        long_text = "x" * (force_threshold + 100) + "。"

        # boundary_detection：一条 oversized UC，未做分裂。
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed": [{"indices": [1]}],
        })
        # OverlongUCSplitSkill 被触发：把原串切成 80/20。
        prefix_len = int((force_threshold + 101) * 0.8)
        fake_llm.push_response({
            "prefix_text": long_text[:prefix_len],
            "tail_text": long_text[prefix_len:],
        })
        # enrichment for the synthesized prefix event
        fake_llm.push_response({"summaries": {"L1": "prefix"}, "roles": []})

        sealed = svc.process_input(long_text)
        assert len(sealed) == 1
        prefix_event = sealed[0]

        ues = meta_repo.get_unclosed_events()
        assert len(ues) == 1
        assert ues[0].split_prefix_event_ids == [prefix_event.event_id]
        # 前缀事件长度占大头，tail UC 的长度应当在阈值内。
        assert ues[0].total_length <= force_threshold
        assert not ues[0].oversized

    def test_remediation_abort_keeps_oversized_uc_not_force_sealed(
        self, metabolism_service_with_repair, fake_llm
    ):
        """评估器失败 + 修复也失败：保留 oversized UC，不进入 seal_event。"""
        svc, meta_repo, event_repo = metabolism_service_with_repair
        force_threshold = int(svc._config.len_msg * svc._config.unclosed_force_ratio)  # noqa: SLF001
        long_text = "x" * (force_threshold + 100) + "。"

        # boundary 返回未分裂
        fake_llm.push_response({
            "completed_events": [],
            "new_unclosed": [{"indices": [1]}],
        })
        # split_skill 主动放弃
        fake_llm.push_response({"abort": True})

        sealed = svc.process_input(long_text)
        # ⚠ 关键：不 force-seal，sealed 为空。
        assert sealed == []

        # 事件库里没有任何基本事件。
        all_events = event_repo.list_all(is_abstract=False)
        assert all_events == []

        # UC 里保留原条并带 oversized=True 标记。
        ues = meta_repo.get_unclosed_events()
        assert len(ues) == 1
        assert ues[0].oversized
        assert ues[0].total_length > force_threshold


# =====================================================================
# Recall: expand split_prefix chain
# =====================================================================

class TestRecallExpandsSplitPrefix:
    def test_prefix_is_pulled_in_when_tail_event_is_recalled(self, config, db, vector_store):
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
        svc = RecallService(config, event_repo, role_repo, vector_store, tri_band=tri_band)

        # 前缀事件（仅靠 split_prefix_event_ids 拉进，不直接命中检索）
        prefix = Event(content_raw="武松在景阳冈下连喝十八碗酒", summaries={"L1": "喝酒"})
        event_repo.save(prefix)
        vector_store.upsert_event_vectors(prefix)

        # 尾部事件（直接被检索命中），声明前缀链。
        tail = Event(
            content_raw="武松过冈遇虎，打死猛虎。",
            summaries={"L1": "打虎收尾"},
            split_prefix_event_ids=[prefix.event_id],
        )
        event_repo.save(tail)
        vector_store.upsert_event_vectors(tail)

        block = svc.build_recall_block("景阳冈打虎")
        ids = [it.event_id for it in block.items]
        # 前缀必须出现；尾部也在；前缀在尾部之前。
        assert prefix.event_id in ids
        assert tail.event_id in ids
        assert ids.index(prefix.event_id) < ids.index(tail.event_id)

    def test_prefix_not_duplicated_when_already_recalled(self, config, db, vector_store):
        event_repo = EventRepository(db)
        role_repo = RoleRepository(db)
        tri_band = TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo)
        svc = RecallService(config, event_repo, role_repo, vector_store, tri_band=tri_band)

        prefix = Event(content_raw="prefix event text", summaries={"L1": "prefix"})
        event_repo.save(prefix)
        vector_store.upsert_event_vectors(prefix)

        tail = Event(
            content_raw="tail event text",
            summaries={"L1": "tail"},
            split_prefix_event_ids=[prefix.event_id],
        )
        event_repo.save(tail)
        vector_store.upsert_event_vectors(tail)

        block = svc.build_recall_block("prefix tail")
        ids = [it.event_id for it in block.items]
        assert ids.count(prefix.event_id) == 1
