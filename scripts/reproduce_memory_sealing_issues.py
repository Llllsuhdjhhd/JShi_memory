"""Offline reproduction of duplicate sealing control flow, with a recording sink."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rems.config import REMSConfig
from rems.services.event_service import EventService
from rems.services.metabolism_service import MetabolismService
from rems.skills.boundary_detection import BoundaryDetectionSkill


def main():
    text = "Melanie went running."
    parser = object.__new__(BoundaryDetectionSkill)
    parsed = parser.parse_response({"events": [{"indices": [1], "continues": None}],
        "residual": []}, [text], shadow_count=0, formed_indices=set())
    line = SimpleNamespace(id="UC-example", merged_content=text, total_length=len(text),
        buffer_items=[], interlocutor_attributions=[], split_prefix_event_ids=[])
    outputs = {}
    for mode in ("rejudge_abandoned", "force_save_all"):
        recorded = []

        def seal(content, **kwargs):
            recorded.append(content)
            return SimpleNamespace(event_id=f"event-{len(recorded)}")

        service = object.__new__(MetabolismService)
        service._boundary = SimpleNamespace(detect=lambda *a, **k: parsed)
        service._event_svc = SimpleNamespace(seal_event=seal)
        service._repo = SimpleNamespace(delete_unclosed_event=lambda *a: None,
                                       update_shadow=lambda *a: None)
        service._raise_dormant_same_object = lambda *a: None
        arguments = dict(memory=None, input_id=None, known_roles_hint=None, pre_role_entries=None)
        if mode == "rejudge_abandoned":
            service._rejudge_abandoned([line], **arguments)
        else:
            service._force_save_all(text, "", [line], **arguments)
        outputs[mode] = {"input": text, "sealed": recorded, "sealed_count": len(recorded)}
    event_service = object.__new__(EventService)
    event_service._config = REMSConfig(_env_file=None, _env_prefix="JSHI_EVAL_UNUSED_")
    outputs["long_summary_expected_budgets"] = event_service._compute_budget(7275).summary_level_budgets
    path = ROOT / "runs/locomo_round2_clean/sealing_reproduction.json"
    path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
