#!/usr/bin/env python3
"""Backfill forgetting_factor for legacy events (design/1010).

存量事件在 arousal→强度映射上线前以 1.0 落库；本脚本只更新
activation_energy > 0 且 forgetting_factor == 1.0 的行，避免覆盖已强化值。

Example::

    python scripts/backfill_forgetting_factor.py --db tests/memory_storage/memory_storage_test.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rems.storage.backfill import backfill_event_forgetting_factors
from rems.storage.database import Database


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill event forgetting_factor from activation_energy"
    )
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument(
        "--gain",
        type=float,
        default=2.0,
        help="arousal->strength gain (default: 2.0, neutral 0.5 -> 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="重置为 activation_energy 映射的初始值（清除 reinforce 累积污染），"
        "覆盖所有 forgetting_factor != 初始值 的行；默认只补 factor==1.0 的存量行",
    )
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    database = Database(f"sqlite:///{db_path.as_posix()}")
    if args.reset:
        from rems.storage.backfill import reset_event_forgetting_factors

        n = reset_event_forgetting_factors(
            database, gain=args.gain, dry_run=args.dry_run
        )
    else:
        n = backfill_event_forgetting_factors(
            database, gain=args.gain, dry_run=args.dry_run
        )
    print(
        f"[{'dry-run' if args.dry_run else 'updated'}] {n} rows "
        f"{'would be' if args.dry_run else ''} touched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
