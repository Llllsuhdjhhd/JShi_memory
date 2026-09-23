import pytest
import json
import os
import shutil
from typing import Any
from rems.config import REMSConfig
from rems.storage.database import Database

@pytest.fixture
def tmp_dir(tmp_path):
    d = tmp_path / "rems_test"
    d.mkdir()
    yield str(d)

@pytest.fixture
def config(tmp_dir):
    cfg = REMSConfig()
    cfg.storage.database_url = f"sqlite:///{tmp_dir}/test.db"
    cfg.storage.qdrant_url = ":memory:"
    cfg.embedding.provider = "hash"  # Use deterministic hash to avoid downloading models
    cfg.recall_query_compress_enabled = False
    cfg.ingest_llm_session_enabled = False
    # 生产默认约 900 字才立即封存。既有单测用短文本，并断言当轮封存。
    cfg.event_min_chars = 0
    return cfg

@pytest.fixture
def config_local(tmp_dir):
    """Optional fixture for slow tests that need real sentence-transformers."""
    cfg = REMSConfig()
    cfg.storage.database_url = f"sqlite:///{tmp_dir}/test_local.db"
    cfg.storage.qdrant_url = ":memory:"
    cfg.embedding.provider = "local"
    return cfg

@pytest.fixture
def db(config):
    database = Database(config.storage.database_url)
    database.create_tables()
    return database

class FakeLLM:
    """Mock LLM provider for tests."""
    def __init__(self, config=None):
        self.config = config
        self._responses = []
        self.calls: list[dict[str, Any]] = []

    def push_response(self, response: Any):
        self._responses.append(response)

    def complete(self, task_type: str, messages: list[dict], **kwargs) -> str:
        if not self._responses:
            return "Fake response"
        res = self._responses.pop(0)
        return json.dumps(res) if not isinstance(res, str) else res

    def complete_json(self, task_type: str, messages: list[dict], **kwargs) -> dict[str, Any]:
        self.calls.append({"task_type": task_type, "messages": list(messages)})
        if not self._responses:
            return {}
        res = self._responses.pop(0)
        if isinstance(res, dict):
            return res
        return json.loads(res)

    def complete_json_continue(
        self,
        task_type: str,
        messages: list[dict],
        new_user_content: str,
        **kwargs,
    ) -> tuple[dict[str, Any], list[dict]]:
        messages.append({"role": "user", "content": new_user_content})
        raw = self.complete(task_type, messages, **kwargs)
        messages.append({"role": "assistant", "content": raw})
        res = json.loads(raw) if isinstance(raw, str) else raw
        return res, messages

@pytest.fixture
def fake_llm(config):
    return FakeLLM(config)

class FakeEmbeddingFunction:
    """Mock embedding function for tests."""
    def __call__(self, input):
        return [[0.1] * 128 for _ in input]
    def embed_query(self, text):
        return [0.1] * 128


@pytest.fixture
def vector_store(config, db):
    # 新架构不再使用旧向量库（design/810：向量占位；回忆索引走 recall_pipeline）。
    return None
