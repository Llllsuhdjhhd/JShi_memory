"""Raw indexing must preserve details independently of returned summary levels."""
from rems.models.event import Event
from rems.recall import HashEmbedding, QdrantRecallVectorStore, RecallPipeline
from rems.storage.repository import EventRepository


class RecordingEmbedding(HashEmbedding):
    def __init__(self):
        super().__init__()
        self.documents = []

    def split_document(self, text):
        return [text[i:i + 40] for i in range(0, len(text), 40)]

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return super().embed_documents(texts)


def test_raw_chunks_reconcile_and_summary_changes_do_not_reembed(db):
    repo = EventRepository(db)
    embedding = RecordingEmbedding()
    store = QdrantRecallVectorStore("raw-test", url=":memory:")
    pipe = RecallPipeline(embedding, store, repo, event_index_level="L0")
    raw = "Caroline plays Bach and Mozart. " * 5
    event = Event(subject_id="stone", content_raw=raw, summaries={"L1": "Music."})
    repo.save(event)
    written = pipe.index_event(event)
    assert written > 1
    assert "".join(embedding.documents) == raw.strip()
    assert store.get(event.event_id)["representation_level"] == "L0"
    assert pipe.index_event(event) == 0
    event.summaries = {"L1": "Different short summary."}
    assert pipe.index_event(event) == 0
    # A raw change must update its vector and remove obsolete tail chunks.
    event.content_raw = "Caroline plays Mozart."
    assert pipe.index_event(event) == 1
    assert store.list_event_logical_ids(event.event_id) == [event.event_id]
    event.summaries = {}
    repo.save(event)
    assert pipe.index_event(event) == 0  # Raw indexing does not require an L1.
    store._client.close()


def test_multiple_chunk_hits_use_maximum_not_last_score(db):
    repo = EventRepository(db)
    store = QdrantRecallVectorStore("raw-max-test", url=":memory:")
    pipe = RecallPipeline(HashEmbedding(), store, repo, event_index_level="L0")
    hits = [dict(score=score, payload={"event_id": "e", "type": "event"})
            for score in (0.9, 0.4)]
    store.search = lambda *args, **kwargs: hits
    assert pipe._semantic_signals("stone", "Mozart", (), None)[("event", "e", "")] == 0.9
    store._client.close()


def test_tokenizer_chunks_cover_full_text_without_exceeding_encoder_limit():
    from types import SimpleNamespace
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from rems.recall.providers import SentenceTransformerEmbedding

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    embedding = SentenceTransformerEmbedding("test-local")
    embedding._model = SimpleNamespace(tokenizer=tokenizer, max_seq_length=12)
    text = " ".join(f"detail{i}" for i in range(53)) + " last detail."
    chunks = embedding.split_document(text)
    assert len(chunks) > 1
    covered = set()
    for chunk in chunks:
        assert len(tokenizer(chunk)["input_ids"]) <= 12
        start = text.index(chunk)
        covered.update(range(start, start + len(chunk)))
    assert covered == set(range(len(text)))
