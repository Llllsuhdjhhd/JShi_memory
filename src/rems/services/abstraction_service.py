from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

# 抽象事件：唯一触发路径 = 回忆块 event_id 集合中的频繁子集挖掘（白皮书 §3.2）。
# - 子集大小 >= abstract_subset_min_size（默认 6，可配置）
# - 支持度（被多少条回忆块整体覆盖） >= abstract_subset_min_support（默认 12）
# 满足两条件的**所有**频繁项集（非仅极大）进入候选队列；成功合成一次后即基于最新 ``recall_log``
# 重新挖掘（见 ``abstract_mining_max_refresh_rounds``），避免在已过时的支持度上继续消费子集。

from ..config import REMSConfig
from ..models.event import CompressionBudget, Event
from ..observability import PerfMonitor
from ..strategies.abstraction import AbstractionEvidencePolicy, LeafContentRawEvidencePolicy
from ..skills.event_enrichment import EventEnrichmentSkill
from ..mining.closed_itemsets import mine_closed_frequent_itemsets
from ..skills.inductive_evolution import InductiveEvolutionSkill, SynthesisOutcome
from ..skills.summary_generation import SummaryGenerationSkill
from ..storage.repository import (
    AbstractedSubsetRepository,
    EventRepository,
    RecallLogRepository,
)
from ..storage.vector_store import VectorStore

from .abstraction_constants import (
    SKIPPED_ABSTRACTION_MINING_NOOP_ID,
    SKIPPED_COHERENCE_FINGERPRINT_ID,
    SKIPPED_SYNTHESIS_FAILURE_ID,
)
from .recall_quality import RecallQualityController

if TYPE_CHECKING:
    from .abstraction_shadowing import AbstractionShadowingService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SubsetSynthesisOutcome:
    abstract_event: Event | None = None
    coherence_rejected: bool = False
    synthesis_failed: bool = False


class AbstractionService:
    """Synthesize abstract events from frequent subsets of recall logs (白皮书 §3.2).

    抽象事件特点（白皮书 §3.1/§3.2）：
        - ``role_list=[]``：抽象事件不登记角色，不写入任何角色白描，也不触发语义卡片；
        - ``summaries``：与基本事件同规则（``SummaryGenerationSkill`` L1…Ln 递归 + 熔断）；
        - 合成输入：始终展开到叶子基本事件，拼接其 ``content_raw`` 与角色线索作为证据；
        - ``insight``：由 ``enable_abstract_insight`` 开关控制，关闭时抽象事件不生成 insight。

    触发是「唯一路径」——由 ``mine_and_synthesize`` 扫描 ``recall_log`` 全量历史，找支持度
    ≥ ``abstract_subset_min_support`` 且大小 ≥ ``abstract_subset_min_size`` 的**全部**频繁子集
    （按规模与支持度排序，优先处理较大子集），逐个经护栏后合成并 ``replace_subset`` 登记替换。

    叙事线近似判重（``enable_narrative_dedup``）默认关闭；开启时与 ``is_fired``、``replace_subset`` 互补。
    """

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        vector_store: VectorStore,
        evolution_skill: InductiveEvolutionSkill,
        summary_skill: SummaryGenerationSkill,
        recall_log_repo: RecallLogRepository,
        abstracted_subset_repo: AbstractedSubsetRepository,
        evidence_policy: AbstractionEvidencePolicy | None = None,
        enrichment_skill: EventEnrichmentSkill | None = None,
        perf_monitor: PerfMonitor | None = None,
        recall_quality: RecallQualityController | None = None,
        asf_service: AbstractionShadowingService | None = None,
    ):
        self._config = config
        self._event_repo = event_repo
        self._vector = vector_store
        self._evolution = evolution_skill
        # 优先走 EventEnrichmentSkill 一次性拿所有层级摘要；保留 summary_skill 作为兜底（旧路径）。
        # P1-8 修复：原实现在 generate() 内部循环调用 N 次 LLM（每层一次），抽象事件成本高。
        self._summary = summary_skill
        self._enrichment = enrichment_skill
        self._recall_log_repo = recall_log_repo
        self._fired_repo = abstracted_subset_repo
        self._evidence_policy = evidence_policy or LeafContentRawEvidencePolicy()
        # 可选 PerfMonitor：mining / 叙事线判重各包一层 timer，
        # 同时让叙事线判重的"比对窗口 K"按系统负载动态收紧（白皮书 §2.3）。
        self._perf = perf_monitor
        self._recall_quality = recall_quality
        self._asf = asf_service

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def mine_and_synthesize(self) -> list[Event]:
        """Scan ``recall_log``; synthesize one abstract event per qualifying frequent subset.

        每次回忆结束后调用一次即可。返回新合成的抽象事件列表（可能为空）。

        护栏与收敛（按调用顺序）：
            1. ``is_fired(subset)`` — 字面子集已处理过则跳过。
            2. ``enable_narrative_dedup`` 为真时 ``_is_narrative_duplicate`` — 叙事线近似去重
               （overlap / novelty 双护栏，含叶子展开；比对窗口 K 随 ``PerfMonitor`` 负载收紧）。
            2.5 叙事整合性前置闸门（``abstract_narrative_coherence_enabled``，默认开）—
               ``_synthesize_from_subset`` 内先做一次 LLM 划分（相干叙事线 vs 无关项）：
               相干数 < ``abstract_narrative_coherence_min_count`` 则 ``coherence_rejected`` 跳过；
               否则只用相干子集合成。rate_a 回灌 ``RecallQualityController`` 的 EMA-a。
            3. 合成成功后 ``replace_subset`` — 把所有**完全包含** S 的 recall_log 行中的 S
               替换为新抽象 id（行级「完全包含」去重，多叙事线共享叶子仍由各自行决定）。
            4. 每次成功合成后立即用最新 ``recall_log`` 重新频繁集挖掘（至多 ``abstract_mining_max_refresh_rounds`` 轮），
               避免沿用过期的支持度统计。
            5. 本轮内已成功抽象条数超过 ``abstract_mining_escalate_min_support_after_consecutive_abstracts`` 后，
               后续刷新轮改用 ``abstract_subset_min_support_escalated`` 作为最小支持度。
        """
        if self._perf is not None:
            with self._perf.timer("abstraction_mining"):
                return self._mine_and_synthesize_inner()
        return self._mine_and_synthesize_inner()

    def _mine_and_synthesize_inner(self) -> list[Event]:
        cfg = self._config
        min_size = max(2, cfg.abstract_subset_min_size)
        min_support = max(2, cfg.abstract_subset_min_support)

        created: list[Event] = []
        max_refresh = max(1, cfg.abstract_mining_max_refresh_rounds)
        abstracts_this_pass = 0

        for _round in range(max_refresh):
            effective_support = min_support
            if abstracts_this_pass > cfg.abstract_mining_escalate_min_support_after_consecutive_abstracts:
                effective_support = max(min_support, max(2, cfg.abstract_subset_min_support_escalated))

            rows = self._recall_log_repo.list_all()
            recent_k = cfg.abstract_mining_recent_k
            if self._perf is not None:
                recent_k = self._perf.adjusted_recent_k(
                    recent_k,
                    cfg.abstract_mining_min_recent_k,
                    channel="abstract",
                )
            transactions = self._recall_log_repo.list_recent_transactions(recent_k)
            tx_filtered = [(t, ts) for t, ts in transactions if len(t) >= min_size]
            if len(tx_filtered) < effective_support:
                break

            candidates = mine_closed_frequent_itemsets(
                tx_filtered,
                min_size=min_size,
                min_support=float(effective_support),
                decay_lambda=cfg.abstract_support_decay_lambda,
            )
            if not candidates:
                break

            progressed = False
            for subset, support in candidates:
                if self._fired_repo.is_fired(subset):
                    continue
                if cfg.enable_narrative_dedup:
                    dup_info = self._is_narrative_duplicate(subset)
                else:
                    dup_info = None
                if dup_info is not None:
                    matched_id, overlap, novelty_abs = dup_info
                    logger.info(
                        "narrative-dup-skip: subset_size=%d support=%d matched=%s overlap=%.2f novelty=%d",
                        len(subset), support, matched_id, overlap, novelty_abs,
                    )
                    self._fired_repo.mark_fired(subset, matched_id)
                    continue

                synth = self._synthesize_from_subset(subset, support)
                if synth.abstract_event is not None:
                    ae = synth.abstract_event
                    created.append(ae)
                    abstracts_this_pass += 1
                    self._fired_repo.mark_fired(subset, ae.event_id)
                    logger.info(
                        "Abstract %s created from mined subset_size=%d (support=%d); "
                        "effective_min_support=%d abstracts_this_pass=%d",
                        ae.event_id,
                        len(subset),
                        support,
                        effective_support,
                        abstracts_this_pass,
                    )
                    progressed = True
                    break
                if synth.coherence_rejected:
                    self._fired_repo.mark_fired(subset, SKIPPED_COHERENCE_FINGERPRINT_ID)
                    continue
                if synth.synthesis_failed:
                    self._fired_repo.mark_fired(subset, SKIPPED_SYNTHESIS_FAILURE_ID)
                    continue
                self._fired_repo.mark_fired(subset, SKIPPED_ABSTRACTION_MINING_NOOP_ID)

            if not progressed:
                break

        return created

    # ------------------------------------------------------------------
    # Narrative-line dedupe (灰区拦截)
    # ------------------------------------------------------------------

    def _is_narrative_duplicate(
        self, subset: frozenset[str],
    ) -> tuple[str, float, int] | None:
        """Return ``(matched_abstract_id, overlap, novelty_abs)`` if subset is dup; else ``None``.

        叙事线相似度判断：
            - 把 ``subset`` 全部展开到叶子（含嵌套抽象事件），得 ``leaves_S``。
            - 对最近 ``K`` 条已有抽象事件 A，把 ``A`` 也展开到叶子 ``leaves_A``，比 overlap/novelty。
            - 命中"高重合 + 新增不显著"则返回；否则 None。

        ``K`` 默认为 ``narrative_dup_compare_recent_k``，PerfMonitor 检测到过载时收紧到
        ``narrative_dup_min_compare_recent_k``，避免在已经卡的算法上再花算力（白皮书 §2.3）。
        """
        if self._perf is not None:
            with self._perf.timer("narrative_dedupe"):
                return self._is_narrative_duplicate_inner(subset)
        return self._is_narrative_duplicate_inner(subset)

    def _is_narrative_duplicate_inner(
        self, subset: frozenset[str],
    ) -> tuple[str, float, int] | None:
        cfg = self._config
        overlap_th = cfg.narrative_dup_overlap_threshold
        novelty_ratio_th = cfg.narrative_dup_novelty_min_ratio
        novelty_abs_th = cfg.narrative_dup_novelty_min_abs

        # 叙事线判重的"比对窗口 K"：默认 narrative_dup_compare_recent_k；过载时按 load_factor 收紧。
        if self._perf is not None:
            recent_k = self._perf.adjusted_recent_k(
                cfg.narrative_dup_compare_recent_k,
                cfg.narrative_dup_min_compare_recent_k,
            )
        else:
            recent_k = cfg.narrative_dup_compare_recent_k

        # 把 subset 展开到叶子（subset 里可能含嵌套抽象事件）。
        leaves_s = _resolve_leaves(self._event_repo, subset)
        if not leaves_s:
            return None

        # 取最近 K 条已合成抽象事件做比对；空库 → 直接返回 None。
        recent_abstracts = self._list_recent_abstracts(recent_k)
        if not recent_abstracts:
            return None

        # 比对：找第一个满足"高重合 + 新增不显著"的抽象。
        # 选第一个就够了——若 subset 跟多条抽象都高度相似，按时间倒序优先匹配最近的，
        # 这样调用方能在 fired_repo 里把这条 subset 归到最相关的抽象名下。
        for abs_evt in recent_abstracts:
            leaves_a = set(self._event_repo.resolve_basic_event_ids(abs_evt.event_id))
            if not leaves_a:
                continue
            inter = leaves_s & leaves_a
            if not inter:
                continue
            overlap = len(inter) / len(leaves_s)
            novelty_abs = len(leaves_s - leaves_a)
            if overlap < overlap_th:
                continue
            # 双护栏：novelty 必须 **既** 低于比例阈值 **又** 低于绝对量阈值，才视为"几乎重复"。
            novelty_ratio = novelty_abs / len(leaves_s)
            if novelty_ratio >= novelty_ratio_th and novelty_abs >= novelty_abs_th:
                # 新增信息显著，放过——这是"老素材上的新叙事"。
                continue
            return (abs_evt.event_id, overlap, novelty_abs)
        return None

    def _list_recent_abstracts(self, k: int) -> list[Event]:
        """Return up to ``k`` most recently created abstract events (newest first)."""
        if k <= 0:
            return []
        all_abs = self._event_repo.list_all(is_abstract=True)
        all_abs.sort(key=lambda e: e.create_time, reverse=True)
        return all_abs[:k]

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize_from_subset(self, subset: frozenset[str], support: int) -> SubsetSynthesisOutcome:
        events: list[Event] = []
        for eid in subset:
            e = self._event_repo.get(eid)
            if e is None:
                logger.debug("abstract mining: skip missing event %s", eid)
                continue
            if e.is_tombstoned:
                continue
            events.append(e)
        if len(events) < self._config.abstract_subset_min_size:
            logger.debug(
                "abstract mining: subset shrunk below min_size after DB filtering (%d < %d)",
                len(events), self._config.abstract_subset_min_size,
            )
            return SubsetSynthesisOutcome(None, False)

        events.sort(key=lambda e: (e.create_time, e.event_id))
        cfg = self._config

        # 叙事整合性前置闸门（白皮书 §3.2「相关率 a」）：在送入合成前，先用一次 LLM 把挖矿
        # 子集划分为「相干叙事线」与「无关项」。相干数低于 ``abstract_narrative_coherence_min_count``
        # 则跳过本次抽象；否则只用相干子集做后续合成（丢弃无关项，避免噪声污染抽象证据）。
        # rate_a（相干占比）回灌 RecallQualityController 的 EMA-a 通道，参与动态语义距离钳制。
        coherent_events = events
        if cfg.abstract_narrative_coherence_enabled:
            coh = self._evolution.evaluate_narrative_coherence(events)
            if self._recall_quality is not None:
                self._recall_quality.record_abstraction_coherence_rate(coh.rate_a)
            if not coh.passed_gate:
                logger.info(
                    "coherence gate rejected: subset_size=%d coherent=%d rate_a=%.2f (min=%d)",
                    len(events),
                    len(coh.coherent_event_ids),
                    coh.rate_a,
                    max(1, cfg.abstract_narrative_coherence_min_count),
                )
                return SubsetSynthesisOutcome(None, coherence_rejected=True)
            coherent_events = [e for e in events if e.event_id in coh.coherent_event_ids]
            if len(coherent_events) < len(events):
                logger.info(
                    "coherence gate pruned subset: %d → %d coherent (rate_a=%.2f)",
                    len(events), len(coherent_events), coh.rate_a,
                )

        evidence_events = self._evidence_policy.collect(coherent_events, self._event_repo)
        if not evidence_events:
            logger.debug(
                "abstract mining: no basic content_raw evidence after coherence filtering",
            )
            return SubsetSynthesisOutcome(None, False)

        max_level = max((e.abstraction_level or 0) for e in coherent_events) + 1
        try:
            outcome: SynthesisOutcome = self._evolution.synthesize(
                coherent_events,
                abstraction_level=max_level,
                evidence_events=evidence_events,
            )
            if outcome.rejected_none:
                return SubsetSynthesisOutcome(None, True)
            abstract_event = outcome.event
            if abstract_event is None:
                return SubsetSynthesisOutcome(None, False, synthesis_failed=True)

            # 抽象事件的摘要与基本事件同规则（白皮书 §1.1.3）。
            # 优先一次 LLM 调用拿全部层级（EventEnrichmentSkill, skip_roles=True），与基本事件流共用 prompt 与解析；
            # 没注入 enrichment 时退回 SummaryGenerationSkill 的递归实现（旧路径，多次 LLM 往返）。
            if self._enrichment is not None:
                budget = self._estimate_budget(abstract_event.content_raw)
                er = self._enrichment.enrich(
                    abstract_event.content_raw,
                    budget=budget,
                    skip_roles=True,
                )
                abstract_event.summaries = er.summaries
                abstract_event.summary_lengths = er.summary_lengths
                abstract_event.actual_max_level = er.actual_max_level
            else:
                sr = self._summary.generate(abstract_event.content_raw)
                abstract_event.summaries = sr.summaries
                abstract_event.summary_lengths = sr.summary_lengths
                abstract_event.actual_max_level = sr.actual_max_level

            self._event_repo.save(abstract_event)
            self._index_abstract(abstract_event)

            if self._asf is not None:
                self._asf.propagate(abstract_event, outcome.shadow_lambda)

            for evt in coherent_events:
                if not evt.is_abstract:
                    self._event_repo.update_status(evt.event_id, is_abstracted=True)

            self._apply_abstract_coverage_for_subset(
                frozenset(e.event_id for e in coherent_events),
            )

            rewrote = self._recall_log_repo.replace_subset(
                {e.event_id for e in coherent_events},
                abstract_event.event_id,
            )
            logger.info(
                "recall_log replace_subset rewrote=%d rows (abstract=%s)",
                rewrote,
                abstract_event.event_id,
            )

            return SubsetSynthesisOutcome(abstract_event, False)
        except Exception as exc:
            logger.warning(
                "abstract synthesis failed subset_size=%d support=%d: %s",
                len(subset),
                support,
                exc,
            )
            return SubsetSynthesisOutcome(None, False, synthesis_failed=True)

    def _apply_abstract_coverage_for_subset(self, member_ids: frozenset[str]) -> None:
        """对已Persist的新抽象的成员集：基本事件 + 子抽象节点及其叶子一次性累加 abstract_coverage。"""
        cfg = self._config
        if cfg.abstract_coverage_strength <= 0:
            return
        coverage_ids: set[str] = set()
        for eid in member_ids:
            e = self._event_repo.get(eid)
            if e is None or e.is_tombstoned:
                continue
            if e.is_abstract:
                coverage_ids.add(eid)
                for lid in self._event_repo.resolve_basic_event_ids(eid):
                    coverage_ids.add(lid)
            else:
                coverage_ids.add(eid)
        if not coverage_ids:
            return
        unit = cfg.abstract_coverage_strength / (1.0 + math.log(len(coverage_ids)))
        self._event_repo.apply_abstract_coverage_increment(
            coverage_ids,
            unit,
            cfg.abstract_coverage_max,
        )

    def _estimate_budget(self, content: str) -> CompressionBudget:
        """Compute a coarse summary budget for abstract event content_raw.

        抽象事件摘要复用基本事件 enrichment prompt，必须给出 ``summary_level_budgets``。
        与 ``EventService._compute_budget`` 保持口径一致：L1 = ``raw_len * compression_target_ratio``，
        L2..L10 按 ``summary_decay_factor`` 指数衰减，至 ``summary_fuse_min_chars`` 熔断。
        抽象事件不需要 snapshot/wp/decoration 预算，留空字典。
        """
        cfg = self._config
        raw_len = max(1, len(content))
        l1 = max(int(raw_len * cfg.compression_target_ratio), cfg.summary_fuse_min_chars)
        summary_budgets: dict[str, int] = {"L1": l1}
        for i in range(2, 11):
            prev = summary_budgets[f"L{i-1}"]
            nxt = max(int(prev * cfg.summary_decay_factor), cfg.summary_fuse_min_chars)
            summary_budgets[f"L{i}"] = nxt
        return CompressionBudget(
            raw_len=raw_len,
            total_budget=l1,
            summary_level_budgets=summary_budgets,
            snapshot_level_budgets={},
            wp_budget_per_role=0,
            decoration_budget=0,
            role_count_estimate=0,
        )

    def _index_abstract(self, event: Event) -> None:
        if hasattr(self._vector, "upsert_event_vectors"):
            self._vector.upsert_event_vectors(event)
        else:
            text = event.summaries.get(event.mid_summary_key, event.content_raw)
            self._vector.add_event(event.event_id, text, {"is_abstract": True})


# =====================================================================
# Helpers
# =====================================================================


def _resolve_leaves(event_repo: EventRepository, ids: "frozenset[str] | set[str]") -> set[str]:
    """Expand any abstract events in ``ids`` to their basic-event leaves; basic events pass through.

    抽象事件可作为更高阶抽象的成员；判重时必须先把"含抽象 id 的 subset"统一展开到叶子层
    再比对，否则同一条叙事的高阶抽象 vs 低阶抽象会被误判成"无交集"。复用
    ``EventRepository.resolve_basic_event_ids``（白皮书 §1.1.8 已有 BFS 实现）。
    """
    leaves: set[str] = set()
    for eid in ids:
        for leaf_id in event_repo.resolve_basic_event_ids(eid):
            leaves.add(leaf_id)
    return leaves


# =====================================================================
# Frequent subset mining (pragmatic, closure-style; all frequent, not only maximal)
# =====================================================================


def _find_all_frequent_subsets(
    transactions: list[frozenset[str]],
    min_size: int,
    min_support: int,
) -> list[tuple[frozenset[str], int]]:
    """Return ``[(subset, support)]`` for every frequent itemset meeting thresholds.

    Pragmatic closure enumeration:
        1. Seed candidates with pairwise intersections of size >= min_size.
        2. Iteratively intersect candidates with each transaction to discover
           smaller closures (also ``>= min_size``).
        3. For each discovered closure C compute support = |{T : C ⊆ T}|;
           keep C if support >= min_support.
        4. Return **all** such C sorted by (-|C|, -support, sorted ids) so larger
           supersets tend to be synthesized before their subsets in one pass.

    The algorithm is **closure-complete** for the common case and O(n²·|avg|)
    per iteration; adequate for the scales we expect (<~1k recalls).
    """
    n = len(transactions)
    if n < min_support:
        return []

    seen: set[frozenset[str]] = set()
    queue: list[frozenset[str]] = []

    # Seed with pairwise intersections.
    for i in range(n):
        for j in range(i + 1, n):
            inter = transactions[i] & transactions[j]
            if len(inter) >= min_size and inter not in seen:
                seen.add(inter)
                queue.append(inter)

    if not queue:
        return []

    frequent: dict[frozenset[str], int] = {}

    # BFS closure expansion.
    idx = 0
    while idx < len(queue):
        C = queue[idx]
        idx += 1

        support = sum(1 for T in transactions if C.issubset(T))
        if support >= min_support and len(C) >= min_size:
            frequent[C] = support

        if len(C) <= min_size:
            continue
        # Further shrinking only makes sense if support already meets the floor.
        if support < min_support:
            continue
        for T in transactions:
            if C.issubset(T):
                continue
            sub = C & T
            if len(sub) >= min_size and sub not in seen:
                seen.add(sub)
                queue.append(sub)

    if not frequent:
        return []

    pairs: list[tuple[frozenset[str], int]] = [(C, frequent[C]) for C in frequent]

    def _sort_key(item: tuple[frozenset[str], int]) -> tuple[int, int, tuple[str, ...]]:
        subset, sup = item
        return (-len(subset), -sup, tuple(sorted(subset)))

    pairs.sort(key=_sort_key)
    return pairs
