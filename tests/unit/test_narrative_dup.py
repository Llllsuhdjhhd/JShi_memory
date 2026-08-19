"""Unit tests for AbstractionService._is_narrative_duplicate.

直接测叙事线判重逻辑（白皮书 §3.2 灰区拦截器）：
overlap >= 阈值 AND NOT（novelty 比例 >= 阈值 AND novelty 绝对量 >= 阈值）→ 判定为同一叙事线再次浮现。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from rems.config import REMSConfig
from rems.models.event import Event
from rems.observability import PerfMonitor
from rems.services.abstraction_service import AbstractionService, _find_all_frequent_subsets
from rems.skills.inductive_evolution import InductiveEvolutionSkill
from rems.skills.summary_generation import SummaryGenerationSkill
from rems.storage.database import Database
from rems.embedding.tri_band import TriBandEncoder
from rems.storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    RecallLogRepository,
    RoleRepository,
)
from rems.storage.vector_store import VectorStore

from ..conftest import FakeLLM


@pytest.fixture()
def env(config: REMSConfig, db: Database, fake_llm: FakeLLM, vector_store):
    # 把阈值调到便于小子集触发：
    #   overlap >= 0.6 视为高度重合；novelty 同时满足比例 >= 0.1 与绝对量 >= 2 才算"显著新增"。
    config.narrative_dup_overlap_threshold = 0.6
    config.narrative_dup_novelty_min_ratio = 0.1
    config.narrative_dup_novelty_min_abs = 2
    config.abstract_subset_min_size = 3
    config.abstract_subset_min_support = 3
    config.enable_narrative_dedup = True
    config.abstract_narrative_coherence_enabled = False

    event_repo = EventRepository(db)
    role_repo = RoleRepository(db)
    vector_store.set_tri_band(TriBandEncoder(config, event_repo=event_repo, role_repo=role_repo))
    recall_log_repo = RecallLogRepository(db)
    fired_repo = AbstractedSubsetRepository(db)
    evolution_skill = InductiveEvolutionSkill(fake_llm, config)
    summary_skill = SummaryGenerationSkill(fake_llm, config)

    svc = AbstractionService(
        config,
        event_repo,
        vector_store,
        evolution_skill,
        summary_skill,
        recall_log_repo,
        fired_repo,
    )
    return svc, event_repo, vector_store, recall_log_repo, fake_llm, config


def _save_basic(event_repo: EventRepository, content: str) -> Event:
    e = Event(content_raw=content, summaries={"L1": content[:30]})
    event_repo.save(e)
    return e


def _save_abstract(
    event_repo: EventRepository,
    leaf_ids: list[str],
    *,
    content: str = "abstract",
    create_time: datetime | None = None,
) -> Event:
    e = Event(
        content_raw=content,
        summaries={"L1": content},
        is_abstract=True,
        source_events=list(leaf_ids),
        abstraction_level=1,
    )
    if create_time is not None:
        e.create_time = create_time
    event_repo.save(e)
    return e


class TestIsNarrativeDuplicate:
    def test_returns_none_when_no_existing_abstracts(self, env):
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        sub = frozenset(e.event_id for e in leaves)
        assert svc._is_narrative_duplicate(sub) is None

    def test_exact_overlap_no_novelty_is_dup(self, env):
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]
        a1 = _save_abstract(event_repo, leaf_ids)

        # 完全相同的 subset 再次被挖出 → 必然命中
        sub = frozenset(leaf_ids)
        result = svc._is_narrative_duplicate(sub)
        assert result is not None
        matched_id, overlap, novelty = result
        assert matched_id == a1.event_id
        assert overlap == 1.0
        assert novelty == 0

    def test_high_overlap_low_novelty_abs_is_dup(self, env):
        # overlap 满足，但 novelty 绝对量 < 2 → 仍判为同叙事
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]
        a1 = _save_abstract(event_repo, leaf_ids)

        # 新候选 = {e0, e1, e_new}: overlap=2/3=0.67 >= 0.6, novelty_abs=1 < 2 → 判为 dup
        e_new = _save_basic(event_repo, "e_new")
        sub = frozenset([leaf_ids[0], leaf_ids[1], e_new.event_id])
        result = svc._is_narrative_duplicate(sub)
        assert result is not None
        assert result[0] == a1.event_id

    def test_significant_novelty_passes_through(self, env):
        # overlap 高但同时满足 novelty_ratio>=0.1 与 novelty_abs>=2 → 视作"老素材新叙事"，放过
        svc, event_repo, *_ = env
        # 抽象 A 覆盖 5 个叶子；新候选共享 3 个，引入 2 个新叶子 → 5/5 vs 5 ↑
        # 让 A 有 5 个，subset 有 5 个：overlap=3/5=0.6 ✓，novelty=2 → 满足放过条件
        old_leaves = [_save_basic(event_repo, f"e{i}") for i in range(5)]
        old_ids = [e.event_id for e in old_leaves]
        _save_abstract(event_repo, old_ids)

        new_leaves = [_save_basic(event_repo, f"new{i}") for i in range(2)]
        new_ids = [e.event_id for e in new_leaves]

        sub = frozenset(old_ids[:3] + new_ids)
        # overlap = 3/5 = 0.6, novelty_ratio = 2/5 = 0.4, novelty_abs = 2 → 都满足"显著新增"
        result = svc._is_narrative_duplicate(sub)
        assert result is None

    def test_low_overlap_passes_through(self, env):
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]
        _save_abstract(event_repo, leaf_ids)

        # 候选只跟 A 共享 1/4，overlap=0.25 < 0.6 → 直接放过
        new_leaves = [_save_basic(event_repo, f"new{i}") for i in range(3)]
        sub = frozenset([leaf_ids[0]] + [n.event_id for n in new_leaves])
        assert svc._is_narrative_duplicate(sub) is None

    def test_subset_with_nested_abstract_is_expanded(self, env):
        # subset 里包含一个抽象 id A1，应被展开到 A1 的叶子再做比对
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]
        a1 = _save_abstract(event_repo, leaf_ids)

        # A2 的 leaves 与 A1 的 leaves 完全一致
        # 候选 subset = {A1}（即抽象 id 自身），展开后 = {e0, e1, e2}
        # 跟 A1 自己的叶子集合比 → overlap=1.0, novelty=0 → 应判 dup（matched A1）
        sub = frozenset([a1.event_id])
        result = svc._is_narrative_duplicate(sub)
        assert result is not None
        assert result[0] == a1.event_id

    def test_recent_k_picks_newest_first(self, env):
        # 当多个抽象都能匹配 subset 时，按最近 K 的"newest first"挑第一个返回
        svc, event_repo, *_ = env
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]
        # 老抽象（更早）
        a_old = _save_abstract(
            event_repo, leaf_ids, create_time=datetime.now() - timedelta(days=10),
        )
        # 新抽象（更晚）
        a_new = _save_abstract(
            event_repo, leaf_ids, create_time=datetime.now(),
        )
        sub = frozenset(leaf_ids)
        result = svc._is_narrative_duplicate(sub)
        assert result is not None
        # newest-first → a_new 应优先匹配
        assert result[0] == a_new.event_id
        assert a_old.event_id != result[0]

    def test_perf_monitor_shrinks_recent_k_under_load(self, env):
        # PerfMonitor 过载时收紧比对窗口 K——窗口缩小到只看最近 1 条抽象，老抽象就不再被比对
        svc, event_repo, *_, config = env
        config.narrative_dup_compare_recent_k = 50
        config.narrative_dup_min_compare_recent_k = 1
        leaves = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in leaves]

        a_old = _save_abstract(
            event_repo, leaf_ids, create_time=datetime.now() - timedelta(days=10),
        )
        # newest 抽象不与 sub 重合（用别的叶子），sub 实际上只跟 a_old 重合
        other = [_save_basic(event_repo, f"o{i}") for i in range(3)]
        a_new = _save_abstract(
            event_repo, [o.event_id for o in other], create_time=datetime.now(),
        )

        # 注入一个过载的 PerfMonitor → adjusted_recent_k = 1，只看 a_new，错过 a_old
        pm = PerfMonitor(
            tolerances_ms={"abstraction_mining": 100.0, "narrative_dedupe": 100.0},
            load_factor_max=4.0,
        )
        for _ in range(5):
            pm.record("narrative_dedupe", 1000.0)  # 远超阈值 → 满载
        svc._perf = pm

        sub = frozenset(leaf_ids)
        # 满载下 K=1 → 仅跟 a_new（最新）比对，a_new 与 sub 无交集 → 不判 dup
        assert svc._is_narrative_duplicate(sub) is None

        # 关闭 perf 或恢复正常 → 能看见 a_old → 判 dup
        svc._perf = None
        result = svc._is_narrative_duplicate(sub)
        assert result is not None
        assert result[0] == a_old.event_id


class TestAllFrequentSubsets:
    """非极大：若大小子集同时满足支持度，应全部进入候选列表。"""

    def test_returns_superset_and_subset_when_both_frequent(self):
        big = frozenset({"e1", "e2", "e3", "e4", "e5", "e6"})
        small = frozenset({"e1", "e2", "e3", "e4"})
        transactions = [big] * 4 + [small] * 2
        out = _find_all_frequent_subsets(transactions, min_size=3, min_support=4)
        subsets = {s for s, _ in out}
        assert big in subsets
        assert small in subsets
        by_id = {s: sup for s, sup in out}
        assert by_id[big] == 4
        assert by_id[small] == 6


class TestMineWithDedupe:
    """End-to-end: 已有抽象时，相同子集再次被挖出应被叙事线判重拦截。"""

    def test_mine_skips_already_covered_subset(self, env):
        svc, event_repo, vector_store, recall_log_repo, fake_llm, _ = env
        # 三个叶子事件
        events = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in events]
        for eid in leaf_ids:
            ev = event_repo.get(eid)
            if ev:
                vector_store.upsert_event_vectors(ev)

        # 预先存在一个抽象事件 A1 覆盖这三个叶子
        _save_abstract(event_repo, leaf_ids)

        # recall_log 上同一组合反复被回忆 3 次，使其满足支持度
        for k in range(3):
            recall_log_repo.append(recall_id=f"RCL-{k}", event_ids=leaf_ids)

        # 不应产生新抽象（被叙事线判重拦截）
        created = svc.mine_and_synthesize()
        assert created == []

    def test_mine_with_dedup_off_synthesizes_despite_similar_abstract(self, env):
        """默认产品语义下 narrative dedup 关闭；显式关闭时不应拦近似叙事线。"""
        svc, event_repo, vector_store, recall_log_repo, fake_llm, config = env
        config.enable_narrative_dedup = False

        events = [_save_basic(event_repo, f"e{i}") for i in range(3)]
        leaf_ids = [e.event_id for e in events]
        for eid in leaf_ids:
            ev = event_repo.get(eid)
            if ev:
                vector_store.upsert_event_vectors(ev)

        _save_abstract(event_repo, leaf_ids)

        for k in range(3):
            recall_log_repo.append(recall_id=f"RCL-{k}", event_ids=leaf_ids)

        fake_llm.push_response({
            "cognitive_relation": "CAUSALITY",
            "shadow_lambda": 0.5,
            "content_raw": "new abstract",
            "decoration": None,
        })
        fake_llm.push_response({"summary": "s", "char_count": 10})

        created = svc.mine_and_synthesize()
        assert len(created) >= 1
        assert all(e.is_abstract for e in created)
