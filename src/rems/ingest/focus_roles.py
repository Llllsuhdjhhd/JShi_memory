"""Heuristic focus-role detection before recall (no LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

from ..utils.zh_normalize import normalize_for_substring_match, to_simplified

if TYPE_CHECKING:
    from ..models.role import Role


class RoleReader(Protocol):
    def list_all(self) -> list[Role]:
        ...


# 原文称谓 / 省略名 → 角色库 canonical ``name``（简体，与 LLM 登记一致）
NARRATIVE_NAME_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("林妹妹", "黛玉"),
    ("林姑娘", "黛玉"),
    ("林姑老爷", "林如海"),
    ("宝二爷", "宝玉"),
    ("凤哥儿", "凤姐"),
    ("琏二爷", "贾琏"),
    ("蓉哥儿", "贾蓉"),
    ("珍大爷", "贾珍"),
)


def build_focus_match_corpus(
    raw_input: str = "",
    shadow_content: str = "",
    *extra_texts: str,
) -> str:
    """Single normalized haystack from ingest text + optional act_query etc."""
    parts = [shadow_content, raw_input, *extra_texts]
    return normalize_for_substring_match(" ".join(p for p in parts if p))


def _name_in_corpus(name: str, corpus: str) -> bool:
    needle = normalize_for_substring_match(name)
    return bool(needle) and needle in corpus


def match_focus_role_ids(
    role_repo: RoleReader,
    *,
    raw_input: str = "",
    shadow_content: str = "",
    extra_texts: Sequence[str] = (),
) -> set[str]:
    """Return role IDs whose name/alias appears in the combined ingest text.

    Applies simplified-Chinese normalization so classical (繁体) source text
    matches simplified role registry names. ``extra_texts`` typically includes
    the LLM ``act_query``, which is already simplified but may name roles
    omitted from raw courtesy forms (e.g. 黛玉 vs 林妹妹).
    """
    corpus = build_focus_match_corpus(raw_input, shadow_content, *extra_texts)
    if not corpus.strip():
        return set()

    roles = list(role_repo.list_all())
    by_name: dict[str, str] = {}
    for role in roles:
        key = to_simplified(role.name).strip()
        if key:
            by_name[key] = role.role_id

    focus: set[str] = set()

    for role in roles:
        names = [role.name, *(role.aliases or [])]
        names = sorted({n for n in names if n}, key=len, reverse=True)
        if any(_name_in_corpus(n, corpus) for n in names):
            focus.add(role.role_id)

    for trigger, canonical in NARRATIVE_NAME_TRIGGERS:
        if not _name_in_corpus(trigger, corpus):
            continue
        rid = by_name.get(to_simplified(canonical).strip())
        if rid:
            focus.add(rid)

    return focus
