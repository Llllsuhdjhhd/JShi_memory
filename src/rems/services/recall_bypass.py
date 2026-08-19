from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RecallBypassController:
    """Rate-limited deep recall bypass (§4.7.1)."""

    confidence_threshold: float = 0.5
    rate_limit_per_minute: int = 1
    _last_bypass_ts: float = field(default=0.0, init=False)
    _bypass_count_window: int = field(default=0, init=False)
    _window_start: float = field(default=0.0, init=False)

    def should_bypass(self, max_score: float) -> bool:
        if max_score >= self.confidence_threshold:
            return False
        now = time.time()
        if now - self._window_start > 60:
            self._window_start = now
            self._bypass_count_window = 0
        if self._bypass_count_window >= self.rate_limit_per_minute:
            return False
        self._bypass_count_window += 1
        self._last_bypass_ts = now
        return True
