from __future__ import annotations


def reciprocal_rank_fusion(
    channel_ranks: dict[str, dict[str, int]],
    *,
    channels: list[str],
    k: int = 60,
    missing_rank: int | None = None,
) -> dict[str, float]:
    """Fuse per-channel ranks into RRF scores (higher is better).

    ``channel_ranks[ch][event_id]`` is 1-based rank within channel *ch*.
    Events absent from a channel receive ``missing_rank`` (default: max+1 per channel).
    """
    if not channels:
        return {}

    max_by_channel: dict[str, int] = {}
    for ch in channels:
        ranks = channel_ranks.get(ch) or {}
        max_by_channel[ch] = max(ranks.values()) if ranks else 0

    all_ids: set[str] = set()
    for ch in channels:
        all_ids.update((channel_ranks.get(ch) or {}).keys())

    scores: dict[str, float] = {}
    for eid in all_ids:
        total = 0.0
        for ch in channels:
            ranks = channel_ranks.get(ch) or {}
            if eid in ranks:
                r = ranks[eid]
            else:
                fallback = missing_rank
                if fallback is None:
                    fallback = max_by_channel[ch] + 1
                r = fallback
            total += 1.0 / (k + r)
        scores[eid] = total
    return scores


def ranks_from_scores(
    scores: dict[str, float],
    *,
    reverse: bool = True,
) -> dict[str, int]:
    """Convert raw scores to 1-based ranks (best score -> rank 1)."""
    ordered = sorted(scores.keys(), key=lambda eid: scores[eid], reverse=reverse)
    return {eid: idx + 1 for idx, eid in enumerate(ordered)}
