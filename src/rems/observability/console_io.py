"""UTF-8 console setup (Windows cp936 / legacy terminals)."""

from __future__ import annotations

import logging
import os
import sys


def configure_stdio_utf8() -> None:
    """Use UTF-8 for stdout/stderr; on Windows also set console code page to 65001."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    if sys.platform != "win32":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass


def configure_logging_stdout(level: int = logging.INFO) -> None:
    """Route stdlib logging to stdout (avoids PowerShell treating stderr as errors)."""
    root = logging.getLogger()
    fmt = logging.Formatter("%(levelname)s: %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    if root.handlers:
        root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def configure_quiet_run() -> None:
    """Suppress chatty third-party logs during long scenario ingest (HTTP, skills)."""
    configure_logging_stdout(logging.WARNING)
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "openai._base_client",
        "urllib3",
        "rems",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
