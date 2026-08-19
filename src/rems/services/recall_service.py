from __future__ import annotations

import logging
import math
from typing import Any

# 回忆服务：事件流 + 白描流检索、RRF 融合、遗忘/情绪修饰与懒摘要降级。
# 总长约束约 1/6.6 上下文（config.physical_redline）。对照白皮书 4.4。

from ..config import REMSConfig
from ..models.event import Event, EventStatus
from ..models.metabolism import ContextPackage, RecallBlock, RecallItem, Shadow
from ..observability import PerfMonitor
from ..strategies.recall import (
    DefaultRecallScoringStrategy,
    DefaultSummaryTierPolicy,
    RecallScoringStrategy,
    SummaryTierPolicy,
)
from ..strategies.forgetting import DefaultWhitePaintingRetentionStrategy
from ..storage.repository import EventRepository, RoleRepository
from ..embedding.tri_band import TriBandEncoder
from ..embedding.query_act_processor import QueryActProcessor, build_query_act_processor
from ..skills.recall_intent import RecallIntentClassifier
from ..storage.active_pool_cache import ActivePoolCache
from ..strategies.event_weight import EventWeightDeriver
from ..storage.vector_store import VectorStore

from ..retrieval.bm25_index import EventBm25Index
from ..retrieval.channels import tri_band_ranks_from_hits
from ..retrieval.factory import create_recall_backend, resolve_recall_profile
from ..retrieval.protocol import RecallBackend
from ..retrieval.rrf import reciprocal_rank_fusion
from ..retrieval.types import RecallContext
from .recall_bypass import RecallBypassController
from .recall_quality import RecallQualityController

logger = logging.getLogger(__name__)


class RecallService:
    """Builds the recall block for each processing cycle.

    Steps:
        1. Semantic search via vector store.
        2. Retrieve full events and compute hybrid relevance scores
           (cosine + time-decay + role-importance + AE affective energy).
        3. Apply Lazy Index degradation (L1 -> mid -> max summary level).
        4. Append role semantic cards for top-scoring roles.
        5. Enforce 1/6.6 context ceiling via ultra-concise mapping if necessary.

    为每次处理周期构造回忆块与上下文包。步骤与英文一致：先向量检索；再取回事件并计算混合相关度
    （余弦相似度、时间半衰衰减、角色重要性、事件级 AE）；在长度预算下做 L1→中间档→更高级摘要的
    「懒索引」降级；若仍有空间则为高分相关角色附加语义卡片；若总长度仍逼近约 1/6.6 上下文上限，
    则启用 ultra 级极简映射以硬裁剪（白皮书 4.4）。
    """

    def __init__(
        self,
        config: REMSConfig,
        event_repo: EventRepository,
        role_repo: RoleRepository,
        vector_store: VectorStore,
        scoring_strategy: RecallScoringStrategy | None = None,
        summary_tier_policy: SummaryTierPolicy | None = None,
        perf_monitor: PerfMonitor | None = None,
        recall_quality: RecallQualityController | None = None,
        active_pool_cache: ActivePoolCache | None = None,
        tri_band: TriBandEncoder | None = None,
        event_weight: EventWeightDeriver | None = None,
        intent_classifier: RecallIntentClassifier | None = None,
        bypass_controller: RecallBypassController | None = None,
        query_act_processor: QueryActProcessor | None = None,
        recall_backend: RecallBackend | None = None,
        bm25_index: EventBm25Index | None = None,
    ):
        self._config = config
        self._event_repo = event_repo
        self._role_repo = role_repo
        self._vector = vector_store
        self._scoring_strategy = scoring_strategy or DefaultRecallScoringStrategy(config)
        self._summary_tier_policy = summary_tier_policy or DefaultSummaryTierPolicy(config)
        self._perf = perf_monitor
        self._forgetting_strategy = DefaultWhitePaintingRetentionStrategy(config, perf_monitor=perf_monitor)
        self._recall_quality = recall_quality
        self._active_pool = active_pool_cache
        self._tri_band = tri_band
        self._event_weight = event_weight
        self._intent = intent_classifier or RecallIntentClassifier(config)
        self._bypass = bypass_controller or RecallBypassController(
            confidence_threshold=config.lifecycle.bypass_confidence_threshold,
            rate_limit_per_minute=config.lifecycle.bypass_rate_limit_per_minute,
        )
        self._query_act_processor = query_act_processor or build_query_act_processor(config)
        _bm25 = bm25_index or EventBm25Index()
        _bm25.set_role_repo(role_repo)
        self._bm25_index = _bm25
        self._backend = recall_backend or create_recall_backend(
            config, event_repo, role_repo, _bm25,
        )
        self._act_source_cache: dict[str, str] = {}
        self.query_act_audit: dict[str, Any] | None = None
        self.recall_backend_trace: dict[str, Any] | None = None
        # 单次 build_recall_block 内对流 A/B 距离过滤的有效 cap（分位帽 ∩ EMA 帽）；None = 本条不额外收紧。
        self._stream_distance_cap_effective: float | None = None
        # 最近一轮回忆拼装用的分位帽/EMA/effective（供诊断）；每次 _prepare_* 覆盖。
        self.recall_distance_cap_audit_snapshot: dict[str, Any] | None = None

    def refresh_backend(self, mode: str | None = None) -> str:
        """Re-bind RecallBackend from config + optional ProcessingMode (per ingest)."""
        profile = resolve_recall_profile(self._config, mode)
        self._backend = create_recall_backend(
            self._config,
            self._event_repo,
            self._role_repo,
            self._bm25_index,
            mode=mode,
        )
        return profile

    # ------------------------------------------------------------------
    @staticmethod
    def _linear_percentile(samples: list[float], p: float) -> float:
        """``p`` in [0,100]; linear interpolation on sorted samples (nearest at degenerate lengths)."""
        if not samples:
            return float("nan")
        xs = sorted(samples)
        if len(xs) == 1:
            return xs[0]
        p_clamped = max(0.0, min(100.0, float(p)))
        if p_clamped <= 0.0:
            return xs[0]
        if p_clamped >= 100.0:
            return xs[-1]
        rank = (len(xs) - 1) * (p_clamped / 100.0)
        lo = int(math.floor(rank))
        hi = int(math.ceil(rank))
        lo = max(0, min(lo, len(xs) - 1))
        hi = max(0, min(hi, len(xs) - 1))
        if lo == hi:
            return xs[lo]
        w = rank - lo
        return xs[lo] * (1.0 - w) + xs[hi] * w

    def _prepare_stream_distance_cap_from_hits(
        self,
        hits_a: list[dict[str, Any]],
        hits_b: list[dict[str, Any]],
    ) -> None:
        """双流原始 distance 池化后取配置分位数，并与 ``RecallQualityController.semantic_distance_cap`` 取 min。"""
        cfg = self._config
        dists: list[float] = [float(h.get("distance", 1.0)) for h in hits_a]
        dists.extend(float(h.get("distance", 1.0)) for h in hits_b)

        pct_spec = cfg.recall_dynamic_distance_hit_percentile
        percentile_cap: float | None = None
        if pct_spec is not None and dists:
            raw = RecallService._linear_percentile(dists, float(pct_spec))
            if not math.isnan(raw):
                lo, hi = cfg.recall_dynamic_distance_cap_floor, cfg.recall_dynamic_distance_cap_ceiling
                if lo > hi:
                    lo, hi = hi, lo
                percentile_cap = max(lo, min(hi, raw))

        ema_cap: float | None = None
        if self._recall_quality is not None:
            ema_cap = self._recall_quality.semantic_distance_cap()

        eff: float | None
        if percentile_cap is not None and ema_cap is not None:
            eff = min(percentile_cap, ema_cap)
        elif percentile_cap is not None:
            eff = percentile_cap
        else:
            eff = ema_cap

        self._stream_distance_cap_effective = eff
        self.recall_distance_cap_audit_snapshot = {
            "hit_distances_count": len(dists),
            "hit_percentile_config": pct_spec,
            "percentile_cap_after_clamp": percentile_cap,
            "ema_semantic_cap": ema_cap,
            "effective_cap": eff,
        }
        logger.debug("recall stream distance cap audit %s", self.recall_distance_cap_audit_snapshot)

    def _clear_stream_distance_cap_scope(self) -> None:
        self._stream_distance_cap_effective = None

    # ------------------------------------------------------------------
    def build_recall_block(
        self,
        query: str,
        shadow: Shadow | None = None,
        focus_role_ids: set[str] | None = None,
        focus_role_entries: list["EventRoleEntry"] | None = None,
        act_source: str | None = None,
        act_source_label: str = "llm_compress",
        ingest_seq: int | None = None,
    ) -> RecallBlock:
        """Build the recall block for *query*. Multi-stage retrieval implementation (白皮书 4.4)."""
        self._act_source_cache = {}
        self.query_act_audit = {"entries": []}

        search_text = query
        if shadow and shadow.content:
            search_text = shadow.content + "\n" + query

        if act_source is not None:
            act_src = self._seed_act_source(search_text, act_source, source=act_source_label)
        else:
            act_src = self._query_act_source(search_text)

        # ========================================================
        # Collective Forgetting & Lifecycle Maintenance
        # ========================================================
        # 在每次检索前，尝试清理/静默那些已被所有参与角色“集体遗忘”的事件。
        # 实际工程中可异步执行，此处为确保逻辑闭环，对检索到的潜在命中做实时校验。

        # ========================================================
        # Tri-band unified retrieval (§4.4.0)
        # ========================================================
        intent = self._intent.classify(search_text)
        pool_limit = self._config.tier1.active_pool_base
        if self._perf is not None:
            pool_limit = self._perf.adjusted_active_pool_limit(
                self._config.tier1.active_pool_base,
                self._config.tier1.active_pool_min,
            )
        active_ids: list[str] | None = None
        if self._active_pool is not None:
            active_ids = self._active_pool.fetch(pool_limit)

        n_results = self._config.recall_n_results
        hits: list[dict] = []
        if self._tri_band is not None:
            valence = self._current_query_valence(focus_role_entries or [])
            if self._config.recall_multi_route_enabled:
                # 多路召回取并集：当前句 / 上下文（残影截尾）/ 实体锚定，最大化覆盖、避免长残影稀释。
                hits = self._multi_route_search(
                    query, shadow, focus_role_ids or set(),
                    valence=valence, weights=intent.weights,
                    active_ids=active_ids, n_results=n_results,
                    act_source=act_src,
                )
            else:
                query_vec = self._tri_band.encode_query(
                    search_text, valence=valence, act_source=act_src,
                )
                hits = self._vector.search_tri_band(
                    query_vec,
                    weights=intent.weights,
                    filter_ids=active_ids,
                    n_results=n_results,
                )
            # 命中过弱时绕过活跃池过滤兜底一次，再与已有命中并集。
            max_score = max((h.get("score", 0.0) for h in hits), default=0.0)
            if self._bypass.should_bypass(max_score):
                query_vec = self._tri_band.encode_query(
                    search_text, valence=valence, act_source=act_src,
                )
                bypass_hits = self._vector.search_tri_band(
                    query_vec,
                    weights=intent.weights,
                    filter_ids=None,
                    n_results=n_results,
                    bypass_filter=True,
                )
                hits = self._union_hits([hits, bypass_hits])
        else:
            hits = self._vector.search(search_text, n_results=n_results)

        stream: dict[str, tuple[Event, float, float]] = {}
        for hit in hits:
            eid = hit.get("event_id")
            if not eid:
                continue
            event = self._event_repo.get(eid)
            if event is None or event.is_tombstoned or event.status.value == "silent":
                continue
            if self._check_and_silence_event(event):
                continue
            distance = float(hit.get("distance", 1.0))
            eff_w = 0.0
            if self._event_weight is not None:
                eff_w = self._event_weight.derive_wi(event)
            stream[eid] = (event, distance, eff_w)

        recall_ctx = RecallContext(
            query=query,
            shadow=shadow,
            act_source=act_src,
            search_text=search_text,
            focus_role_ids=focus_role_ids or set(),
            focus_role_entries=focus_role_entries or [],
            active_event_ids=active_ids,
            ingest_seq=ingest_seq,
        )
        tri_ranks = tri_band_ranks_from_hits(hits)
        channel_ranks: dict[str, dict[str, int]] = {"tri_band": tri_ranks}
        channel_ranks.update(
            self._backend.auxiliary_channel_ranks(
                recall_ctx,
                tri_band_event_ids=list(stream.keys()),
            )
        )
        self.recall_backend_trace = self._backend.on_recall_trace()
        if self.recall_backend_trace is not None:
            self.recall_backend_trace["channel_sizes"] = {
                ch: len(ranks) for ch, ranks in channel_ranks.items()
            }

        scored_events = self._multichannel_stream_score(
            stream, channel_ranks, focus_role_entries or [],
        )
        decay = self._config.abstract_coverage_decay_rate
        if decay > 0:
            scored_events = [
                (ev, sc * self._recall_coverage_weight(ev, decay))
                for ev, sc in scored_events
            ]
            scored_events.sort(key=lambda x: x[1], reverse=True)

        # Apply intermediate filter target (1.2/6.6)
        redline = self._config.physical_redline
        intermediate_ceiling = int(redline * self._config.recall_intermediate_filter_factor)
        
        filtered_events: list[tuple[Event, float]] = []
        current_len = 0
        for event, score in scored_events:
            tier_offset = self._compute_tier_offset(event, focus_role_ids or set())
            text, _ = self._role_aware_pick_summary(event, 9999, tier_offset)
            if not text:
                continue
            
            # Integrity: Stop BEFORE the item that would exceed the intermediate ceiling
            if current_len + len(text) > intermediate_ceiling and filtered_events:
                break
                
            filtered_events.append((event, score))
            current_len += len(text)

        block = self._assemble_block(filtered_events, focus_role_ids=focus_role_ids or set())

        # 80/20 分裂前缀展开（2026-05）：硬性把命中事件的前缀链拉进回忆块。
        # 允许降档压缩，但不允许丢弃；超长时兜底走 ultra-concise fallback 也保留 event_id。
        block = self._expand_split_prefixes(block, focus_role_ids=focus_role_ids or set())

        # 恢复逻辑（Recovery Logic）: 当事件被实际回忆（即包含在 block 中），
        # 加强该事件包含的所有角色白描的遗忘因子，实现回忆后记忆加强。
        for item in block.items:
            event = self._event_repo.get(item.event_id)
            if not event:
                continue
            for role_entry in event.role_list:
                wp_entry = self._role_repo.get_white_painting_by_event(role_entry.role_id, event.event_id)
                if wp_entry:
                    new_factor = min(
                        max(wp_entry.forgetting_factor, 1.0) * self._config.recall_reinforce_multiplier,
                        self._config.recall_forgetting_factor_cap,
                    )
                    self._role_repo.update_white_painting_access(role_entry.role_id, event.event_id, new_factor)

        return block

    # ------------------------------------------------------------------
    # Multi-route candidate generation (本轮新增：覆盖最大化)
    # ------------------------------------------------------------------
    def _multi_route_search(
        self,
        query: str,
        shadow: Shadow | None,
        focus_role_ids: set[str],
        *,
        valence: float,
        weights: tuple[float, float, float],
        active_ids: list[str] | None,
        n_results: int,
        act_source: str,
    ) -> list[dict[str, Any]]:
        """Union of complementary query routes to raise recall coverage.

        三路（每路独立检索后按 event_id 取最小 distance 并集）：
            - 路 C（当前句）：仅当前输入，保证当前话题主信号不被残影稀释；
            - 路 S（上下文）：残影（按 ``recall_query_shadow_cap`` 取尾部）+ 当前输入，保叙事连续；
            - 路 E（实体锚定）：以焦点角色名/别名作为 ``entity_hint`` 偏置实体带，保底拉取角色相关事件。
        下游打分 / RRF / 压缩完全不变——并集只增加候选，不改变排序与组装逻辑。
        """
        encode = self._tri_band.encode_query
        hit_lists: list[list[dict[str, Any]]] = []

        # 路 C：当前句（act 与 search_text 压缩结果共用，ent/emo 仍用各路 query 文本）
        hit_lists.append(
            self._vector.search_tri_band(
                encode(
                    query,
                    valence=valence,
                    act_source=act_source,
                ),
                weights=weights, filter_ids=active_ids, n_results=n_results,
            )
        )

        # 路 S：上下文（残影截尾 + 当前句）
        shadow_text = shadow.content if (shadow and shadow.content) else ""
        if shadow_text:
            cap = self._config.recall_query_shadow_cap
            if cap and cap > 0 and len(shadow_text) > cap:
                shadow_text = shadow_text[-cap:]
            ctx_text = (shadow_text + "\n" + query).strip()
            if ctx_text and ctx_text != query:
                hit_lists.append(
                    self._vector.search_tri_band(
                        encode(
                            ctx_text,
                            valence=valence,
                            act_source=act_source,
                        ),
                        weights=weights, filter_ids=active_ids, n_results=n_results,
                    )
                )

        # 路 E：实体锚定
        entity_hint = self._build_entity_hint(focus_role_ids)
        if entity_hint:
            hit_lists.append(
                self._vector.search_tri_band(
                    encode(
                        query,
                        valence=valence,
                        entity_hint=entity_hint,
                        act_source=act_source,
                    ),
                    weights=weights, filter_ids=active_ids, n_results=n_results,
                )
            )

        return self._union_hits(hit_lists)

    def _seed_act_source(
        self,
        text: str,
        act_text: str,
        *,
        source: str = "llm_compress",
    ) -> str:
        """Pre-compute act embed text (e.g. from ingest session Turn1); skip LLM in processor."""
        key = (text or "").strip()
        self._act_source_cache[key] = act_text
        if self.query_act_audit is not None:
            self.query_act_audit["entries"].append({
                "input_chars": len(key),
                "act_text": act_text[:200],
                "source": source,
            })
        return act_text

    def _query_act_source(self, text: str) -> str:
        cached = self._act_source_cache.get(text)
        if cached is not None:
            return cached
        result = self._query_act_processor.resolve_act_text(text)
        self._act_source_cache[text] = result.act_text
        if self.query_act_audit is not None:
            self.query_act_audit["entries"].append({
                "input_chars": len(text or ""),
                "act_text": result.act_text[:200],
                "source": result.source,
            })
        return result.act_text

    def _build_entity_hint(self, focus_role_ids: set[str]) -> str:
        """Join focus roles' names + aliases as an entity-band hint string."""
        if not focus_role_ids:
            return ""
        names: list[str] = []
        for rid in focus_role_ids:
            role = self._role_repo.get(rid)
            if role is None:
                continue
            if role.name:
                names.append(role.name)
            names.extend(a for a in (role.aliases or []) if a)
        return " ".join(dict.fromkeys(names))  # 去重保序

    @staticmethod
    def _union_hits(hit_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Union hits across routes, keeping the closest (min distance) hit per event_id."""
        best: dict[str, dict[str, Any]] = {}
        for hits in hit_lists:
            for h in hits:
                eid = h.get("event_id")
                if not eid:
                    continue
                prev = best.get(eid)
                if prev is None or float(h.get("distance", 1.0)) < float(prev.get("distance", 1.0)):
                    best[eid] = h
        return list(best.values())

    def _multichannel_stream_score(
        self,
        stream: dict[str, tuple[Event, float, float]],
        channel_ranks: dict[str, dict[str, int]],
        focus_role_entries: list["EventRoleEntry"],
    ) -> list[tuple[Event, float]]:
        """Multi-channel RRF + Factor/Mood modifiers (§4.4 + RagProfile channels)."""
        channels = list(self._backend.rrf_channels)
        if "tri_band" not in channels:
            channels = ["tri_band", *channels]
        rrf_scores = reciprocal_rank_fusion(
            channel_ranks,
            channels=channels,
            k=self._config.recall_rrf_k,
        )
        current_valence = self._current_query_valence(focus_role_entries)
        merged: list[tuple[Event, float]] = []
        for eid, (event, _dist, eff_w) in stream.items():
            rrf = rrf_scores.get(eid, 0.0)
            if rrf <= 0:
                continue
            factor_modifier = 1.0 + self._config.recall_factor_alpha * math.log10(
                1.0 + max(eff_w, 0.0)
            )
            mood_modifier = self._mood_modifier(event.event_valence, current_valence)
            merged.append((event, rrf * factor_modifier * mood_modifier))
        return sorted(merged, key=lambda x: x[1], reverse=True)[:60]

    def _single_stream_score(
        self,
        stream: dict[str, tuple[Event, float, float]],
        focus_role_entries: list["EventRoleEntry"],
    ) -> list[tuple[Event, float]]:
        """Legacy single-stream rank (tests / fallback)."""
        sorted_items = sorted(stream.items(), key=lambda x: x[1][1])
        rank_map = {eid: idx + 1 for idx, (eid, _) in enumerate(sorted_items)}
        return self._multichannel_stream_score(
            stream, {"tri_band": rank_map}, focus_role_entries,
        )

    def _rrf_merge(
        self,
        stream_a: dict[str, tuple[Event, float]],
        stream_b: dict[str, tuple[Event, float, float]],
        focus_role_entries: list["EventRoleEntry"],
    ) -> list[tuple[Event, float]]:
        """Reciprocal-Rank-Fusion + Modifier 融合（白皮书 §4.4）。

        阶段 1: 各流按"距离升序"独立产出 Rank（流 A 用原始余弦距离，流 B 用按遗忘因子
        修正后的有效距离）。
        阶段 2:
            ``Final = (1/(k+R_A) + 1/(k+R_B)) * Factor_Modifier * Mood_Modifier``
            - Factor_Modifier 仅作用于流 B 命中（焦点角色对该事件的关注强度）；
            - Mood_Modifier 在情绪同号时放大，异号时不变（避免负向召回过激）。
        """
        # 距离升序 → Rank 1 = 距离最近 = 语义最相关
        sorted_a = sorted(stream_a.items(), key=lambda x: x[1][1])
        sorted_b = sorted(stream_b.items(), key=lambda x: x[1][1])
        rank_a = {eid: idx + 1 for idx, (eid, _) in enumerate(sorted_a)}
        rank_b = {eid: idx + 1 for idx, (eid, _) in enumerate(sorted_b)}
        event_ids = set(rank_a) | set(rank_b)
        current_valence = self._current_query_valence(focus_role_entries)
        k = self._config.recall_rrf_k

        merged: list[tuple[Event, float]] = []
        for eid in event_ids:
            event = stream_a[eid][0] if eid in stream_a else stream_b[eid][0]
            rrf = 0.0
            if eid in rank_a:
                rrf += 1.0 / (k + rank_a[eid])
            if eid in rank_b:
                rrf += 1.0 / (k + rank_b[eid])

            # 角色重要性 / 关注度调制：只在流 B 命中（被某个焦点角色"反向激活"）时启用。
            if eid in stream_b:
                effective_forgetting = stream_b[eid][2]
                factor_modifier = 1.0 + self._config.recall_factor_alpha * math.log10(
                    1.0 + max(effective_forgetting, 0.0)
                )
            else:
                factor_modifier = 1.0

            mood_modifier = self._mood_modifier(event.event_valence, current_valence)
            merged.append((event, rrf * factor_modifier * mood_modifier))

        return sorted(merged, key=lambda x: x[1], reverse=True)[:60]

    @staticmethod
    def _recall_coverage_weight(event: Event, decay_rate: float) -> float:
        """RRF 分乘子：coverage 越大权重越小，恒为正。"""
        c = float(getattr(event, "abstract_coverage", 0.0) or 0.0)
        if c <= 0:
            return 1.0
        return math.exp(-decay_rate * c)

    def _current_query_valence(self, focus_role_entries: list["EventRoleEntry"]) -> float:
        if not focus_role_entries:
            return 0.0
        return sum(e.emotional_model.valence for e in focus_role_entries) / len(focus_role_entries)

    def _mood_modifier(self, event_valence: float, current_valence: float) -> float:
        if event_valence == 0.0 or current_valence == 0.0:
            return 1.0
        if event_valence * current_valence <= 0:
            return 1.0
        return 1.0 + self._config.recall_mood_beta * (event_valence * current_valence)

    # ------------------------------------------------------------------
    def build_context_package(
        self,
        raw_input: str,
        shadow: Shadow,
        focus_role_ids: set[str] | None = None,
        focus_role_entries: list["EventRoleEntry"] | None = None,
        act_source: str | None = None,
        act_source_label: str = "llm_compress",
        ingest_seq: int | None = None,
    ) -> ContextPackage:
        # 整段回忆块组装（含双流检索 + RRF + 修饰 + 懒索引降级）的耗时由 PerfMonitor 监控；
        # 越线即抬高 load_factor，进而让 forgetting silence 阈值收紧、抽象判重 K 收窄。
        block_kwargs = {
            "focus_role_ids": focus_role_ids,
            "focus_role_entries": focus_role_entries,
            "act_source": act_source,
            "act_source_label": act_source_label,
            "ingest_seq": ingest_seq,
        }
        if self._perf is not None:
            with self._perf.timer("recall_assembly"):
                recall = self.build_recall_block(raw_input, shadow, **block_kwargs)
        else:
            recall = self.build_recall_block(raw_input, shadow, **block_kwargs)
        return ContextPackage(
            recall_block=recall,
            shadow=shadow,
            current_input=raw_input,
        )

    # ------------------------------------------------------------------
    # Legacy hook（白皮书 §4.4 改造前的中间打分，主回路已切到 RRF + Modifier）
    # ------------------------------------------------------------------

    def _hybrid_score(self, event: Event, cosine_sim: float) -> float:
        """Legacy mixed score, kept only for diagnostic tracers.

        新版主回路（``build_recall_block`` → ``_rrf_merge``）只用"距离 → 名次 → RRF"
        + Factor / Mood Modifier，不再混合此函数的绝对量打分（白皮书 §4.4 摒弃绝对量
        融合的设计原则）。该方法保留是为旧观测脚本（recall scoring tracer 等）
        提供历史兼容入口。
        """
        return self._scoring_strategy.score(event, cosine_sim).total

    def _check_and_silence_event(self, event: Event) -> bool:
        """Check if all roles in the event have 'forgotten' it.
        
        如果事件关联的所有角色的遗忘因子均低于阈值，则将其设为 SILENT 并返回 True。
        """
        if not event.role_list:
            return False
            
        threshold = self._config.event_silence_threshold
        all_forgot = True
        for re in event.role_list:
            wp = self._role_repo.get_white_painting_by_event(re.role_id, event.event_id)
            if wp:
                f_score = self._forgetting_strategy.score(wp, is_penalized=True)
                if not f_score.is_silenced and f_score.effective_forgetting >= threshold:
                    all_forgot = False
                    break
            else:
                # 缺失白描条目的角色视为已遗忘
                continue
        
        if all_forgot:
            logger.info("Event %s silenced: all %d roles have forgotten it", event.event_id, len(event.role_list))
            self._event_repo.update_status(event.event_id, status=EventStatus.SILENT)
            return True
        return False

    def _filter_hits_semantic_distance(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop vector hits whose distance exceeds the active cap (分位帽∩EMA 或其一)."""

        if not hits:
            return hits
        cap = self._stream_distance_cap_effective
        if cap is None and self._recall_quality is not None:
            cap = self._recall_quality.semantic_distance_cap()
        if cap is None:
            return hits
        return [h for h in hits if float(h.get("distance", 1.0)) <= cap + 1e-9]

    def _get_stream_a_hits(self, query: str, focus_role_ids: set[str]) -> list[dict]:
        """Tiered Stream A retrieval implementation (70/30 Rule, 白皮书 §4.4)."""
        capacity = self._config.recall_max_capacity
        global_ratio = self._config.recall_global_ratio

        total_count = self._vector.count()
        if total_count <= capacity:
            # 库容量未达上限，执行标准全局检索（不施加 status 过滤——历史索引可能没写 status 字段）。
            return self._vector.search(query, n_results=60)

        # 超过容量，计算 70% 边界时刻（基于 SQL 真实活跃事件总数）
        boundary_dt = self._event_repo.find_time_boundary(capacity, global_ratio)
        if not boundary_dt:
            return self._vector.search(query, n_results=60)

        boundary_ts = boundary_dt.timestamp()

        # 分层检索：
        # 1. 最近 global_ratio 区域（全局可见）
        hits_recent = self._vector.search(
            query,
            n_results=60,
            where={"create_time": {"$gte": boundary_ts}},
        )

        # 2. 剩余 1-global_ratio 区域（仅焦点角色关联可见）
        hits_older: list[dict] = []
        if focus_role_ids:
            # 改用每角色一位标志位 ``role_<id>: True``——这是 Chroma metadata 唯一支持的多值过滤方式。
            # 旧实现用 ``role_ids:{$contains:rid}`` 在真 Chroma 后端无效（$contains 仅作用于 documents）。
            or_filters = [{f"role_{rid}": True} for rid in focus_role_ids]
            if or_filters:
                role_clause = or_filters[0] if len(or_filters) == 1 else {"$or": or_filters}
                hits_older = self._vector.search(
                    query,
                    n_results=40,
                    where={"$and": [
                        {"create_time": {"$lt": boundary_ts}},
                        role_clause,
                    ]},
                )

        # 合并并去重
        seen: set[str] = set()
        merged: list[dict] = []
        for h in hits_recent + hits_older:
            if h["event_id"] not in seen:
                merged.append(h)
                seen.add(h["event_id"])

        # 按距离截断前 60
        merged.sort(key=lambda x: x.get("distance", 1.0))
        return merged[:60]

    def _get_stream_b_hits(self, query: str) -> list[dict]:
        """Stream B: White-painting (Role) semantic search."""
        return self._vector.search_white_paintings(query, n_results=60)

    @staticmethod
    def _time_decay(create_time, half_life_days: float = 30.0) -> float:
        """Backward-compatible wrapper for the default recall time decay."""
        return DefaultRecallScoringStrategy.time_decay(create_time, half_life_days)

    # ------------------------------------------------------------------
    # Assembly with role-aware Lazy Index degradation
    # ------------------------------------------------------------------

    def _assemble_block(
        self,
        scored: list[tuple[Event, float]],
        focus_role_ids: set[str] | None = None,
    ) -> RecallBlock:
        """Assemble RecallBlock with Sliding Window Dynamic Compression (白皮书 4.4 演化).
        
        Logic:
        1. Split scored events into Head (Top 2/3) and Tail (Bottom 1/3).
        2. Iteratively compress the Tail group by increasing summary tier offsets.
        3. If Tail is fully compressed, shift items from Head into Tail and repeat.
        """
        focus_role_ids = focus_role_ids or set()
        ceiling = self._config.physical_redline
        head_ratio = self._config.recall_head_ratio
        
        # Initial categorization
        n = len(scored)
        head_count = int(n * head_ratio)
        
        # We store the base tier offset and current 'extra' compression for each item
        items_data = []
        for i, (event, score) in enumerate(scored):
            base_offset = self._compute_tier_offset(event, focus_role_ids)
            items_data.append({
                "event": event,
                "score": score,
                "base_offset": base_offset,
                "extra_compression": 0,
                "is_head": i < head_count
            })

        max_iterations = 20 # Safety break
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            # 1. Trial assembly
            current_items = []
            total_len = 0
            for data in items_data:
                offset = data["base_offset"] + data["extra_compression"]
                text, level = self._role_aware_pick_summary(data["event"], 9999, offset)
                if not text: continue
                
                # Integrity check for final assemble: stop before crossing redline
                if total_len + len(text) > ceiling and current_items:
                    # Keep ``total_len`` as the actual assembled length.
                    # Over-counting here would trigger unnecessary extra compression.
                    break

                current_items.append(RecallItem(
                    event_id=data["event"].event_id,
                    content=text,
                    score=data["score"],
                    summary_level=level
                ))
                total_len += len(text)
            
            # 2. Check if we fit
            if total_len <= ceiling or not items_data:
                break
                
            # 3. Progressive Compression
            # Target the Tail group (including those shifted from Head)
            tail_indices = [i for i, d in enumerate(items_data) if not d["is_head"]]
            
            can_compress_tail = False
            if tail_indices:
                # 判断「还能不能再压一档」：比较当前档与再加一档后选出的摘要。
                # 若选出的 level_key 发生改变且文本确实更短，才允许 extra_compression++；
                # 否则说明已经到了该事件的最高压缩档（``_role_aware_pick_summary`` 的 clamp），
                # 继续递增没有意义，避免空转循环耗尽 ``max_iterations``。
                for idx in tail_indices:
                    event = items_data[idx]["event"]
                    current_level = items_data[idx]["base_offset"] + items_data[idx]["extra_compression"]
                    cur_text, cur_key = self._role_aware_pick_summary(event, 9999, current_level)
                    next_text, next_key = self._role_aware_pick_summary(event, 9999, current_level + 1)
                    if next_key and next_key != cur_key and len(next_text) < len(cur_text):
                        items_data[idx]["extra_compression"] += 1
                        can_compress_tail = True
            
            # 4. Shifting Logic
            if not can_compress_tail:
                # If tail is maxed out, shift one item from Head to Tail
                head_indices = [i for i, d in enumerate(items_data) if d["is_head"]]
                if head_indices:
                    last_head_idx = head_indices[-1]
                    logger.info("Shifting item %s from Head to Tail for compression", items_data[last_head_idx]["event"].event_id)
                    items_data[last_head_idx]["is_head"] = False
                    # On shift, we might want to immediately apply one level of compression
                    items_data[last_head_idx]["extra_compression"] += 1
                else:
                    # Everything is already in Tail and maxed out. 
                    # Use ultra-concise fallback and exit.
                    current_items = self._ultra_concise_fallback(current_items, ceiling)
                    break

        block = RecallBlock(items=current_items)
        block.recompute_length()
        return block

    # ------------------------------------------------------------------
    # Role-aware summary tier helpers
    # ------------------------------------------------------------------

    def _compute_tier_offset(self, event: Event, focus_role_ids: set[str]) -> int:
        """Return the tier offset for *event* given *focus_role_ids*.

        Scans ``event.role_list`` to find the highest importance of any focus
        role.  Maps that importance to an offset relative to the mid-level index:

        * S / A  →  mid - ``recall_primary_role_detail_shift``  (more detail)
        * B      →  mid + ``recall_default_tier_offset``         (default)
        * C / D  →  mid + ``recall_minor_role_compress_shift``   (more compressed)
        * absent →  mid + ``recall_minor_role_compress_shift``   (same as C/D)
        """
        return self._summary_tier_policy.tier_offset(event, focus_role_ids)

    def _role_aware_pick_summary(
        self,
        event: Event,
        budget: int,
        tier_offset: int,
    ) -> tuple[str, str]:
        """Pick a summary level for *event* using role-aware tier selection.

        Algorithm
        ---------
        1. Build the ordered list of available summary levels: [L1, L2, …, Ln].
        2. Compute *target index* = mid_index + tier_offset, clamped to [0, n-1].
        3. Starting at *target index*, search toward max (higher compression) for
           a summary that fits *budget*.
        4. If target summary char-count ≤ ``recall_summary_min_chars``, pin at
           that level — do NOT compress further regardless of tier_offset.
        5. If nothing fits, fall back to L0 (content_raw) or return empty.

        字数门槛（recall_summary_min_chars）保证过短摘要不会被进一步压缩；
        这在摘要条目极短（如只剩几个关键词）时非常重要。
        """
        min_chars = self._config.recall_summary_min_chars

        # No summaries — try raw content
        if not event.summaries:
            text = event.content_raw
            return (text, "L0") if len(text) <= budget else ("", "")

        levels = sorted(event.summaries.keys(), key=lambda k: int(k[1:]))
        n = len(levels)
        mid_idx = n // 2  # index of the middle level (default starting point)

        target_idx = max(0, min(n - 1, mid_idx + tier_offset))

        # Starting from target, scan toward max-compression until we find one
        # that fits budget AND respects the min_chars floor.
        for idx in range(target_idx, n):
            level_key = levels[idx]
            text = event.summaries[level_key]
            char_count = len(text)

            # Min-chars floor: if this summary is already very short,
            # treat it as the terminal level — use it if it fits budget.
            at_floor = char_count <= min_chars

            if char_count <= budget:
                return text, level_key

            if at_floor:
                # Can't compress further; if it doesn't fit budget, give up.
                break

        # If target is above mid (requesting more detail), also scan toward L1.
        if target_idx < mid_idx:
            for idx in range(target_idx, -1, -1):
                level_key = levels[idx]
                text = event.summaries[level_key]
                if len(text) <= budget:
                    return text, level_key

        # Last resort: try every level from most to least compressed
        for level_key in reversed(levels):
            text = event.summaries[level_key]
            if len(text) <= budget:
                return text, level_key

        return "", ""

    # ------------------------------------------------------------------
    # 80/20 split-prefix expansion (2026-05)
    # ------------------------------------------------------------------

    def _expand_split_prefixes(
        self,
        block: RecallBlock,
        focus_role_ids: set[str],
    ) -> RecallBlock:
        """Ensure every prefix in the split chain of a recalled event is co-recalled.

        语义约束：
            1. **硬性拉入**：只要某条被召回事件有 ``split_prefix_event_ids``，
               其前缀事件（含多跳）都必须出现在回忆块中；
            2. **允许降档**：前缀事件初始以"比默认档再压一级"的摘要落块；
            3. **越顶兜底**：总长度越过 ``physical_redline`` 时，对前缀事件走
               ``_ultra_concise_fallback`` 把长度截成 ``[EVT-xxx] <stub>…`` 也要保留；
            4. **防链路爆炸**：前缀链展开深度不超过 ``recall_split_prefix_max_depth``，
               并去重；同一条前缀只入一次；
            5. **严格时间语序**：前缀事件在**对应后继条目之前**出现——若前缀已在 block
               但排在后继之后（RRF 分数偶尔反转），把它**重排**到后继之前。
        """
        if not block.items:
            return block

        max_depth = max(1, int(self._config.recall_split_prefix_max_depth))

        # 1. 扫描 block，收集每条命中事件的完整前缀链（去重、控深度）。
        chains: list[tuple[str, list[Event]]] = []  # (successor_event_id, far-to-near prefixes)
        all_required_prefix_ids: set[str] = set()
        for item in block.items:
            event = self._event_repo.get(item.event_id)
            if event is None or not event.split_prefix_event_ids:
                continue
            prefixes = self._walk_prefix_chain(
                event,
                depth_cap=max_depth,
                visited=set(),
            )
            if prefixes:
                chains.append((item.event_id, prefixes))
                for p in prefixes:
                    all_required_prefix_ids.add(p.event_id)

        if not chains:
            return block

        # 2. 以 event_id → RecallItem 的 map 持有当前 block 状态，
        #    并记录原始 block items 的顺序。新插入的前缀按 chains 顺序排定位置。
        existing_by_id: dict[str, RecallItem] = {it.event_id: it for it in block.items}

        # 3. 为"尚未在 block 中"的前缀事件构造 RecallItem（稍压一档的摘要）。
        for successor_id, prefixes in chains:
            successor_item = existing_by_id.get(successor_id)
            successor_score = successor_item.score if successor_item else 0.0
            for p in prefixes:
                if p.event_id in existing_by_id:
                    continue
                text, level_key = self._role_aware_pick_summary(p, 9999, 1)
                if not text:
                    text = p.content_raw
                    level_key = "L0"
                existing_by_id[p.event_id] = RecallItem(
                    event_id=p.event_id,
                    content=text,
                    score=successor_score * 0.9,
                    summary_level=level_key,
                )

        # 4. 重建 block items，使得每条 chain 的 prefix 按自远而近排在 successor 之前；
        #    其它命中（非 prefix / 非 successor）按原始顺序保留。
        original_ids = [it.event_id for it in block.items]
        emitted: set[str] = set()
        final_items: list[RecallItem] = []

        # 先构建每个事件的"必须出现在它之前"的前缀列表（去重、最后出现的 chain 胜出）。
        prefixes_before: dict[str, list[str]] = {}
        for successor_id, prefixes in chains:
            prefixes_before[successor_id] = [p.event_id for p in prefixes]

        def _emit(eid: str) -> None:
            if eid in emitted:
                return
            # 若有前缀依赖，先递归 emit 它们。
            for pid in prefixes_before.get(eid, []):
                _emit(pid)
            emitted.add(eid)
            item = existing_by_id.get(eid)
            if item is not None:
                final_items.append(item)

        # 先 emit 原始 items 顺序中的每一条；_emit 内部会把前缀事件先入队。
        for eid in original_ids:
            _emit(eid)

        # 5. 超顶兜底：只对新加入 / 非锚定的前缀事件做 ultra-concise 截断。
        ceiling = self._config.physical_redline
        total = sum(len(it.content) for it in final_items)
        if total > ceiling:
            # "锚定"指原始 block items；它们不参与降级。前缀事件允许降到 ultra。
            anchor_ids = set(original_ids)
            for i, it in enumerate(final_items):
                if total <= ceiling:
                    break
                if it.event_id in anchor_ids:
                    continue
                if it.summary_level == "ultra":
                    continue
                stub = it.content[:40] + "…"
                concise = f"[{it.event_id}] {stub}"
                if len(concise) < len(it.content):
                    total -= (len(it.content) - len(concise))
                    final_items[i] = RecallItem(
                        event_id=it.event_id,
                        content=concise,
                        score=it.score,
                        summary_level="ultra",
                    )

        out = RecallBlock(items=final_items)
        out.recompute_length()
        return out

    def _walk_prefix_chain(
        self,
        event: Event,
        *,
        depth_cap: int,
        visited: set[str],
    ) -> list[Event]:
        """Return prefix events in far-to-near order, dedup & cycle-safe.

        ``event.split_prefix_event_ids`` 本身就是"自远而近"顺序；如果某个前缀自己
        还有前缀链（多跳），在它自己位置之前先展开自己的前缀，保持时间语序稳定。
        """

        out: list[Event] = []

        def _recurse(pid: str, depth: int) -> None:
            if depth > depth_cap or pid in visited:
                return
            visited.add(pid)
            prefix_evt = self._event_repo.get(pid)
            if prefix_evt is None or prefix_evt.is_tombstoned:
                return
            if prefix_evt.status.value == "silent":
                return
            # 先递归展开该前缀**自己**的链（自远而近），再把它自己追加。
            for grand_pid in prefix_evt.split_prefix_event_ids or []:
                _recurse(grand_pid, depth + 1)
            out.append(prefix_evt)

        for pid in event.split_prefix_event_ids or []:
            _recurse(pid, 1)
        return out

    @staticmethod
    def _ultra_concise_fallback(items: list[RecallItem], ceiling: int) -> list[RecallItem]:
        """Compress tail items to ``event_id + core verb + role_ids`` triples.

        当组装后总长仍越过上限时，对放不下的条目将 ``content`` 截断为前缀加省略号，并包以 ``[event_id]`` 前缀，
        标记 ``summary_level=ultra``，以在极端预算下保留可追溯 ID 与少量语义（工程化硬裁剪，白皮书 4.4）。
        """
        result: list[RecallItem] = []
        running = 0
        for it in items:
            if running + len(it.content) <= ceiling:
                result.append(it)
                running += len(it.content)
            else:
                stub = it.content[:40] + "…"
                concise = f"[{it.event_id}] {stub}"
                if running + len(concise) <= ceiling:
                    result.append(RecallItem(
                        event_id=it.event_id,
                        content=concise,
                        score=it.score,
                        summary_level="ultra",
                    ))
                    running += len(concise)
        return result
