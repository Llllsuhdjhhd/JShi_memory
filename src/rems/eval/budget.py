"""Persistent reservations shared by every paid benchmark component.

Peak Flash prices, no cache discount. UTF-8 bytes plus framing bound prompt tokens
conservatively for byte-based tokenization. Unknown/failed requests retain their
full reservation: a timeout is not evidence that the provider did not bill it.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path


class BudgetExceeded(RuntimeError):
    pass


class BudgetLedger:
    def __init__(self, path: Path, limit_yuan: float = 90):
        if not 0 < limit_yuan <= 90:
            raise ValueError("Submission limit must be in (0, 90] yuan")
        self.path, self.limit = Path(path), limit_yuan
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextmanager
    def locked(self):
        with self._lock:
            lock = self.path.with_suffix(self.path.suffix + ".lock")
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise RuntimeError("Budget ledger is locked; refuse concurrent untracked spending")
            try:
                yield
            finally:
                os.close(fd)
                lock.unlink()

    def _rows(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text("utf-8").splitlines() if line]

    def _append(self, row):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def summary(self):
        latest = {}
        for row in self._rows():
            latest[row["id"]] = row
        known = sum(r["charge"] for r in latest.values() if r["state"] == "settled")
        held = sum(r["charge"] for r in latest.values() if r["state"] != "settled")
        return {"known_cost_yuan": round(known, 6), "held_yuan": round(held, 6),
                "committed_yuan": known + held, "limit_yuan": self.limit,
                "calls": len(latest), "unsettled": sum(r["state"] != "settled" for r in latest.values())}

    def reserve(self, messages, max_output: int, task: str):
        prompt_bound = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 256
        charge = (prompt_bound * 2 + max_output * 8) / 1_000_000
        with self.locked():
            summary = self.summary()
            if summary["unsettled"]:
                raise BudgetExceeded("Unsettled request exists; refuse further spending until reconciled")
            if summary["committed_yuan"] + charge > self.limit:
                raise BudgetExceeded(f"Budget would exceed {self.limit} yuan ({task})")
            row = {"id": uuid.uuid4().hex, "state": "reserved", "charge": charge,
                   "task": task, "prompt_bound": prompt_bound, "max_output": max_output}
            self._append(row)
        return row

    def settle(self, reservation, usage):
        if usage is None:
            raise RuntimeError("Missing API usage; reservation retained, stopping evaluation")
        def val(key):
            return usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        prompt, output = val("prompt_tokens"), val("completion_tokens")
        if prompt is None or output is None or prompt < 0 or output < 0:
            raise RuntimeError("Invalid API usage; reservation retained")
        charge = (prompt * 2 + output * 8) / 1_000_000
        with self.locked():
            if any(r["id"] == reservation["id"] and r["state"] == "settled" for r in self._rows()):
                raise RuntimeError("Reservation already settled")
            self._append({**reservation, "state": "settled", "charge": charge,
                          "prompt_tokens": prompt, "completion_tokens": output})
        if charge > reservation["charge"]:
            raise RuntimeError("API usage exceeded conservative reservation; stop and inspect pricing")

    def charge_unknown_at_reserved_maximum(self, reservation_id, reason):
        """Resolve an interrupted call at its full bound, without inventing usage."""
        if not reason.strip():
            raise ValueError("An audit reason is required")
        with self.locked():
            matches = [row for row in self._rows() if row["id"] == reservation_id]
            if not matches or matches[-1]["state"] == "settled":
                raise ValueError("Expected one unresolved reservation")
            self._append({**matches[-1], "state": "settled", "usage_unknown": True,
                          "accounting": "full_reserved_upper_bound", "reason": reason})
