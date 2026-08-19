"""Load Hongloumeng chunks from ``data/hongloumeng_dataset.json``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tests.scenarios.common.harness import find_repo_root


@dataclass(frozen=True)
class Chunk:
    id: int
    content: str
    size: int


def dataset_path(repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    return root / "data" / "hongloumeng_dataset.json"


def load_dataset(repo_root: Path | None = None) -> list[Chunk]:
    path = dataset_path(repo_root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Chunk(id=int(item["id"]), content=str(item["content"]), size=int(item["size"]))
        for item in raw
    ]


def get_chunk(chunk_id: int, repo_root: Path | None = None) -> Chunk:
    for c in load_dataset(repo_root):
        if c.id == chunk_id:
            return c
    raise KeyError(f"Chunk id {chunk_id} not in dataset")


def iter_chunks(
    *,
    start_after_id: int = 0,
    count: int = 1,
    repo_root: Path | None = None,
) -> Iterator[Chunk]:
    """Yield up to *count* chunks with id > start_after_id, in dataset order."""
    yielded = 0
    for c in load_dataset(repo_root):
        if c.id <= start_after_id:
            continue
        yield c
        yielded += 1
        if yielded >= count:
            break
