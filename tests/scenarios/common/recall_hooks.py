"""Tri-band recall tracing hooks (2026.06 — replaces dual-stream A/B)."""

from __future__ import annotations

from typing import Any, Callable

from rems.services.recall_service import RecallService


def _hit_preview(h: dict) -> dict[str, Any]:
    return {
        k: h[k]
        for k in ("event_id", "score", "distance")
        if k in h and h[k] is not None
    }


def install_recall_hooks(svc: RecallService) -> tuple[dict[str, Any], Callable[[], None]]:
    """Wrap tri-band search and capture lightweight snapshots."""

    trace: dict[str, Any] = {
        "tri_band_searches": [],
        "intent": None,
        "bypass": [],
        "query_act": None,
        "recall_backend": None,
    }
    vector = svc._vector
    orig_search = vector.search_tri_band
    intent_classifier = svc._intent
    orig_build = svc.build_recall_block

    def wrap_build(*args, **kwargs):
        block = orig_build(*args, **kwargs)
        trace["query_act"] = svc.query_act_audit
        trace["recall_backend"] = svc.recall_backend_trace
        return block

    orig_classify = intent_classifier.classify

    def wrap_classify(query: str):
        result = orig_classify(query)
        trace["intent"] = {
            "weights": result.weights,
            "label": getattr(result, "label", None),
        }
        return result

    def wrap_search(*args, **kwargs):
        hits = orig_search(*args, **kwargs)
        trace["tri_band_searches"].append({
            "n_hits": len(hits),
            "weights": kwargs.get("weights"),
            "filter_ids_count": len(kwargs.get("filter_ids") or []),
            "bypass_filter": kwargs.get("bypass_filter", False),
            "top_hits": [_hit_preview(h) for h in hits[:20]],
        })
        if kwargs.get("bypass_filter"):
            trace["bypass"].append({"n_hits": len(hits)})
        return hits

    intent_classifier.classify = wrap_classify  # type: ignore[method-assign]
    vector.search_tri_band = wrap_search  # type: ignore[method-assign]
    svc.build_recall_block = wrap_build  # type: ignore[method-assign]

    def uninstall():
        intent_classifier.classify = orig_classify  # type: ignore[method-assign]
        vector.search_tri_band = orig_search  # type: ignore[method-assign]
        svc.build_recall_block = orig_build  # type: ignore[method-assign]

    return trace, uninstall
