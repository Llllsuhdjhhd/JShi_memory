from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

# 角色服务：注册、白描追加、基于 arousal 的白描摘要权重（白皮书第 2 章）。

from ..config import REMSConfig, UserMode
from ..models.event import Event, EventRoleEntry, Importance
from ..models.role import Role, WhitePaintingEntry, generate_role_id
from ..skills.role_extraction import ExtractedRole, RoleExtractionSkill
from ..storage.repository import RoleRepository

logger = logging.getLogger(__name__)

class RoleService:
    """Manages global role registry and the white-painting system.

    维护全局 ``role_id`` 注册、按时间排序的白描流水，以及在后台（本实现为同进程同步调用）
    并提供基于 arousal 的白描摘要加权抽取，体现动态遗忘（白皮书第 2 章）。
    """

    def __init__(
        self,
        config: REMSConfig,
        role_repo: RoleRepository,
        role_skill: RoleExtractionSkill,
        llm: object | None = None,
        vector_store: "VectorStore | None" = None,
    ):
        self._config = config
        self._repo = role_repo
        self._skill = role_skill
        self._vector_store = vector_store

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_role(self, name: str, entity_type: str = "person", aliases: list[str] | None = None, is_suspicious: bool = False) -> Role:
        existing = self._repo.find_by_name(name)
        if existing:
            logger.info("Role '%s' already exists as %s", name, existing.role_id)
            return existing

        role = Role(
            role_id=generate_role_id(),
            name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            is_suspicious=is_suspicious,
        )
        self._repo.save(role)
        logger.info("Registered new role %s (%s)", role.role_id, name)
        return role

    def get_role(self, role_id: str) -> Optional[Role]:
        return self._repo.get(role_id)

    def find_role(self, name: str) -> Optional[Role]:
        return self._repo.find_by_name(name)

    def list_roles(self) -> list[Role]:
        return self._repo.list_all()

    # ------------------------------------------------------------------
    # White-painting updates driven by a sealed event
    # ------------------------------------------------------------------

    def update_from_event(self, event: Event) -> None:
        """For every role referenced in *event*, append a white-painting entry.

        对 *event* 的 ``role_list`` 中每个角色：若库中无该 ``role_id`` 则注册（名称回退为 ID 本身）；
        取 L2 互动白描或 L1 提及作为 ``role_summary``；按角色/事件 AE 计算 ``memory_weight``；
        追加白描条目并触发语义卡片刷新。抽象事件直接跳过（白皮书 2.2、2.3）。
        """
        if event.is_abstract:
            return

        for entry in event.role_list:
            role = self._repo.get(entry.role_id)
            if role is None:
                role = self.register_role(entry.role_id, entity_type="person")

            summary_text = self._pick_wp_summary(entry)

            memory_weight = min(max(entry.emotional_model.arousal, 0.0), 1.0)
            base_forgetting = 100.0 * (memory_weight ** self._config.emotion_arousal_gamma)

            wp = WhitePaintingEntry(
                event_id=event.event_id,
                role_summary=summary_text,
                emotional_model=entry.emotional_model,
                importance=entry.importance,
                create_time=event.create_time,
                memory_weight=memory_weight,
                forgetting_factor=base_forgetting,
                base_forgetting_factor=base_forgetting,
                is_suspicious=role.is_suspicious,
            )
            self._repo.add_white_painting_entry(role.role_id, wp)
            if self._vector_store:
                self._vector_store.add_white_painting(
                    role_id=role.role_id,
                    event_id=event.event_id,
                    text=summary_text,
                    metadata={"forgetting_factor": base_forgetting}
                )
            logger.debug("WP appended for %s from event %s (AE=%.2f)", role.role_id, event.event_id, memory_weight)

    # ------------------------------------------------------------------
    # Dynamic granularity routing — 收集端（白皮书 2.3）
    # ------------------------------------------------------------------

    def _pick_wp_summary(self, entry: EventRoleEntry) -> str:
        """Select the snapshot level to persist into the white-painting timeline.

        选择写入白描时间线的快照粒度：
        - 主要角色（``S``/``A`` 重要性）或单人模式下的核心用户 → 使用 ``wp_primary_field``（默认 L3 决策白描）；
        - 其他角色 → 使用 ``wp_default_field``（默认 L2 互动白描）；
        - 若目标字段为空则按 L3→L2→L1 顺序回退。与《REMS 记忆系统规范解析》2.3 对齐，
          让白描收集端实现"主角详细 / 配角标准"的动态粒度路由，防止长周期性格漂移。
        """
        cfg = self._config
        is_primary = self._is_primary_role(entry)
        target_field = cfg.wp_primary_field if is_primary else cfg.wp_default_field

        snap = entry.role_snapshot
        candidates_primary = [
            target_field,
            "l3_decision",
            "l2_interaction",
            "l1_mention",
        ]
        candidates_default = [
            target_field,
            "l2_interaction",
            "l1_mention",
            "l3_decision",
        ]
        ordered = candidates_primary if is_primary else candidates_default

        seen: set[str] = set()
        for field_name in ordered:
            if field_name in seen:
                continue
            seen.add(field_name)
            text = getattr(snap, field_name, None)
            if text:
                return text
        return ""

    def _is_primary_role(self, entry: EventRoleEntry) -> bool:
        # S/A 重要性 → 主角；单人模式下核心用户恒为主角（白皮书 2.2 / 2.3）。
        imp = entry.importance if isinstance(entry.importance, Importance) else Importance(str(entry.importance))
        if imp in (Importance.S, Importance.A):
            return True
        cfg = self._config
        if cfg.user_mode == UserMode.SINGLE and cfg.core_user_role_id == entry.role_id:
            return True
        return False

    # ------------------------------------------------------------------
    # Dynamic forgetting: weight-based decay for white-painting retrieval
    # ------------------------------------------------------------------

    def get_white_painting_summary(self, role_id: str, *, limit: int = 20) -> str:
        """Return a compact textual summary with capacity-aware soft forgetting.

        容量感知的白描检索（白皮书 2.3 FIFO 软遗忘）：
        1. 取全量白描条目（按 create_time 升序）。
        2. 计算总字符长度。若 ≤ ``wp_role_capacity`` → 全量保真，不施加遗忘惩罚。
        3. 若超出容量 → 从最老条目（FIFO）开始逐条标记为"受惩罚"，直到剩余未惩罚条目
           总长回落至容量上限以内。
        4. 受惩罚条目以时间半衰 + AE 遗忘因子降权；未惩罚条目保持完整权重。
        5. 按综合得分取 top *limit* 条拼成可读摘要。
        """
        # 获取全量白描（按时间升序）
        all_entries = self._repo.get_white_painting(role_id)
        if not all_entries:
            return ""

        capacity = self._config.wp_role_capacity

        # 计算总长并确定 FIFO 惩罚边界
        total_len = sum(len(e.role_summary) for e in all_entries)

        # 找到惩罚边界：从最老开始累计，直到剩余未惩罚条目总长 ≤ capacity
        penalized_count = 0
        if total_len > capacity:
            cumulative = 0
            for i, e in enumerate(all_entries):
                if total_len - cumulative <= capacity:
                    break
                cumulative += len(e.role_summary)
                penalized_count = i + 1

        # 打分
        now = datetime.now()
        scored: list[tuple[WhitePaintingEntry, float]] = []
        for i, e in enumerate(all_entries):
            is_penalized = i < penalized_count

            if is_penalized:
                # 受惩罚条目：施加时间半衰 + AE 遗忘因子
                age_days = max((now - e.create_time).total_seconds() / 86400, 0.0)
                half_life = self._config.wp_half_life_days
                retention = math.exp(-0.693 * age_days / half_life)
                score = e.memory_weight * 0.4 + retention * 0.6
            else:
                # 未惩罚条目：全量保真，完整权重
                score = 1.0

            scored.append((e, score))

        # 按得分降序取 top，再按时间排序输出
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [e for e, _ in scored[:limit]]
        top.sort(key=lambda x: x.create_time)

        lines = [
            f"[{e.create_time.strftime('%Y-%m-%d %H:%M')}] ({e.importance.value if hasattr(e.importance, 'value') else e.importance}) {e.role_summary}"
            for e in top
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Role-aware resolution helpers
    # ------------------------------------------------------------------

    def resolve_and_register(self, extracted: list[ExtractedRole], is_suspicious: bool = False) -> dict[str, str]:
        """Ensure every extracted role has a persistent role_id.

        为抽取结果中每个角色解析或创建持久 ``role_id``：若已带 ID 且库中存在则复用；
        否则按名称查找；再否则 ``register_role`` 新建。返回从抽取键到最终 ``role_id`` 的映射。
        如果本次生成涉及疑难边界仲裁或强制兜底（is_suspicious=True），则新注册角色附带可疑属性。
        """
        mapping: dict[str, str] = {}
        for er in extracted:
            key = er.role_id or er.name
            # If the model returned "null" or some placeholder ROL, treat it as new
            if er.role_id and er.role_id not in ("null", "None", ""):
                role = self._repo.get(er.role_id)
                if role:
                    mapping[key] = role.role_id
                    continue
            
            # Use name-based lookup
            role = self._repo.find_by_name(er.name)
            if role:
                mapping[key] = role.role_id
            else:
                # Sanitize name: if numeric or garbage, don't use it as the definitive name
                clean_name = er.name
                mark_suspicious = is_suspicious
                is_invalid = (
                    er.name.isdigit() or 
                    not er.name.strip() or 
                    er.name.lower() in ("null", "none", "unknown", "核心用户", "未知角色")
                )
                if is_invalid:
                    clean_name = f"未知人物_{generate_role_id()[-4:]}"
                    mark_suspicious = True  # 仲裁边界不定产生垃圾名称，强制标记为可疑
                
                new_role = self.register_role(clean_name, er.entity_type, is_suspicious=mark_suspicious)
                mapping[key] = new_role.role_id
        return mapping
