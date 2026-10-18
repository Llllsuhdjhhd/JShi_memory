"""Run with python -m rems.jobs.refine_memory --subject ID [--watch]."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

from ..config import REMSConfig
from ..pipeline import REMSPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="Defaults to REMS_SUBJECT_ID")
    parser.add_argument("--now", type=datetime.fromisoformat, help="Single-pass historical clock")
    parser.add_argument("--watch", action="store_true", help="Run repeated bounded background passes")
    args = parser.parse_args()
    if args.watch and args.now:
        parser.error("--now is only supported for single-pass replay")
    cfg = REMSConfig()
    subject = args.subject or cfg.subject_id
    if not subject:
        parser.error("--subject or REMS_SUBJECT_ID is required")
    pipe = REMSPipeline.from_config(cfg)
    try:
        while True:
            result = pipe.refine_memory(subject, now=args.now)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["errors"]:
                raise SystemExit(1)
            if not args.watch:
                break
            time.sleep(cfg.memory_refinement.interval_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.recall_pipeline._vector_store._client.close()
        pipe.db.engine.dispose()


if __name__ == "__main__":
    main()
