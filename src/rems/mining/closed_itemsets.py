from __future__ import annotations

import math
from datetime import datetime
_SUPPORT_EPS = 1e-6


def _support_decay(
    transactions: list[tuple[frozenset[str], datetime]],
    subset: frozenset[str],
    now: datetime,
    lam: float,
) -> float:
    total = 0.0
    for txn, ts in transactions:
        if subset.issubset(txn):
            dt_days = max((now - ts).total_seconds() / 86400.0, 0.0)
            total += math.exp(-lam * dt_days)
    return total


def mine_closed_frequent_itemsets(
    transactions: list[tuple[frozenset[str], datetime]],
    *,
    min_size: int,
    min_support: float,
    decay_lambda: float,
    now: datetime | None = None,
) -> list[tuple[frozenset[str], float]]:
    """Closed frequent itemsets with time-decayed support (§3.2.1–3.2.2).

    Pragmatic implementation: enumerate candidates from pairwise intersections,
    keep only closed itemsets (no superset with same support).
    """
    now = now or datetime.now()
    if not transactions:
        return []

    item_support: dict[frozenset[str], float] = {}
    items: set[str] = set()
    for txn, _ in transactions:
        items.update(txn)

    # Seed 1-itemsets
    candidates: set[frozenset[str]] = {frozenset([it]) for it in items}

    # Pairwise intersections for larger candidates
    tx_sets = [t for t, _ in transactions]
    for i in range(len(tx_sets)):
        for j in range(i + 1, len(tx_sets)):
            inter = tx_sets[i] & tx_sets[j]
            if len(inter) >= min_size:
                candidates.add(inter)

    # Expand downward from larger sets
    queue = sorted(candidates, key=len, reverse=True)
    seen: set[frozenset[str]] = set()
    for c in queue:
        if c in seen or len(c) < min_size:
            continue
        seen.add(c)
        sup = _support_decay(transactions, c, now, decay_lambda)
        if sup + _SUPPORT_EPS >= min_support:
            item_support[c] = sup
        if len(c) > min_size:
            for item in c:
                sub = frozenset(x for x in c if x != item)
                if len(sub) >= min_size:
                    seen.add(sub)
                    sup_sub = _support_decay(transactions, sub, now, decay_lambda)
                    if sup_sub + _SUPPORT_EPS >= min_support:
                        item_support[sub] = sup_sub

    # Closed: remove subset if superset has same support
    closed: dict[frozenset[str], float] = {}
    for s, sup_s in item_support.items():
        is_closed = True
        for t, sup_t in item_support.items():
            if s != t and s.issubset(t) and abs(sup_s - sup_t) < 1e-9:
                is_closed = False
                break
        if is_closed:
            closed[s] = sup_s

    pairs = list(closed.items())
    pairs.sort(key=lambda x: (-len(x[0]), -x[1], tuple(sorted(x[0]))))
    return pairs
