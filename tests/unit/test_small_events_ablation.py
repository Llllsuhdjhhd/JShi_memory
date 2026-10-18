import hashlib
import sys
from pathlib import Path

from rems.models.event import Event
from rems.pipeline import REMSPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from locomo_jshi_eval import restore_small_baseline


def test_restore_only_original_small_events_without_mutating_source(config, fake_llm, tmp_path):
    config.memory_refinement.enabled = True
    source = REMSPipeline.from_config(config, llm=fake_llm)
    root = Event(subject_id="stone", memory_stage=0, content_raw="Original precise fact.",
        source_ids=["source-1"], source_fragment_ids=["fragment-1"],
        refinement_objects={"Lee": "lee"}, is_abstracted=True, refinement_hidden_by=["parent"])
    source.event_repo.save(root)
    source.event_repo.save(Event(event_id="parent", subject_id="stone", memory_stage=1,
        content_raw=root.content_raw, summaries={"L1": "Lossy fact."}, source_events=[root.event_id]))
    source_path = Path(config.storage.database_url.removeprefix("sqlite:///"))
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    target_config = config.model_copy(deep=True)
    target_config.storage.database_url = "sqlite:///" + (tmp_path / "target.db").as_posix()
    target = REMSPipeline.from_config(target_config, llm=fake_llm)
    try:
        provenance = restore_small_baseline(source_path, target)
        roots = target.event_repo.list_all()
        assert len(roots) == 1 and roots[0].event_id == root.event_id
        assert roots[0].content_raw == root.content_raw and roots[0].source_fragment_ids == ["fragment-1"]
        assert not roots[0].refinement_hidden_by and not roots[0].is_abstracted
        assert provenance["small_event_count"] == provenance["restored_hidden_sources"] == 1
        assert provenance["excluded_parent_count"] == 1
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == before
        assert not fake_llm.calls
        found = target.recall_pipeline.recall("stone", "precise", channels=("lexical",), reinforce=False)
        assert found and "Original precise fact" in found[0].content
    finally:
        for pipe in (source, target):
            pipe.recall_pipeline._vector_store._client.close()
            pipe.db.engine.dispose()
