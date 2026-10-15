"""LoCoMo 评测运行器（所有对话共用一个实例：专用 SQLite + 专用 Qdrant 路径）。

用法（密钥只从环境变量读，脚本与产物都不记录密钥）::

    REMS_LLM__API_KEY=... python scripts/locomo_eval.py \
        --data /workspace/LoCoMo/locomo10.json --out runs/locomo_smoke \
        --model deepseek-flash --thinking off

所有对话共用一个实例（专用 SQLite + Qdrant 路径）；全局主体 = ``--subject``（默认 protagonist）。
默认只跑第 1 段对话；``--all`` 才跑全部（需评审同意）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rems.config import REMSConfig
from rems.eval import locomo as L
from rems.llm.provider import LLMProvider
from rems.pipeline import REMSPipeline

JUDGE_PROMPT = (
    "Question: {q}\nGold answer: {gold}\nPredicted answer: {pred}\n"
    "Is the predicted answer semantically correct w.r.t. the gold answer? "
    "Reply with exactly CORRECT or WRONG."
)


def _usage(invs):
    return {
        "calls": len(invs),
        "prompt_tokens": sum(i.prompt_tokens or 0 for i in invs),
        "completion_tokens": sum(i.completion_tokens or 0 for i in invs),
        "total_tokens": sum(i.total_tokens or 0 for i in invs),
        "cached_prompt_tokens": sum(getattr(i, "cached_tokens", None) or 0 for i in invs),
        "latency_s": round(sum(i.latency_ms for i in invs) / 1000.0, 2),
        "by_task": {
            t: sum(1 for i in invs if i.task_type == t) for t in sorted({i.task_type for i in invs})
        },
    }


def build_config(args, inst_dir: Path) -> REMSConfig:
    cfg = REMSConfig(
        context_window=args.context_window,
        recall_lexical_latin_mode=args.lexical_latin_mode,
        recall_reinforce_enabled=not args.no_reinforce,
        recall_trace_enabled=True,
        enrichment_perspective=args.perspective,
    )
    # 专用实例：独立 SQLite + 独立 Qdrant 路径（不用默认 rems.db / 内存库）
    cfg.storage.database_url = f"sqlite:///{inst_dir / 'rems.db'}"
    cfg.storage.qdrant_path = str(inst_dir / "qdrant")
    if args.embedding_model:
        cfg.embedding.provider = "local"
        cfg.embedding.model_name = args.embedding_model
    if args.embedding_version is not None:
        cfg.recall_embedding_version = args.embedding_version
    if args.model:  # 显式覆盖全部 task model（含 default）
        for name in type(cfg.llm.task_models).model_fields:
            setattr(cfg.llm.task_models, name, args.model)
    if args.base_url:
        cfg.llm.base_url = args.base_url
    if args.thinking == "off":
        cfg.llm.extra_body = {"thinking": {"type": "disabled"}}
    return cfg


def fmt_memories(frags, max_chars: int) -> str:
    lines = []
    for i, f in enumerate(frags, 1):
        when = f.occurred_at.strftime("%d %B %Y %H:%M") if f.occurred_at else "unknown date"
        lines.append(f"[{i}] ({when}) {(f.content or f.text or '').strip()}")
    return L.cap_context("\n".join(lines), max_chars) or "(none)"


def ingest_sample(pipe, sample, args, subject_id) -> dict:
    sid = sample["sample_id"]
    sessions = sorted(L.iter_sessions(sample), key=lambda s: s.occurred_at)  # 按 session 日期
    log = []
    t0 = time.perf_counter()
    for s in sessions:
        for b in L.session_batches(sample, s, subject_id=subject_id, soft_chars=args.soft_chars,
                                   include_image_caption=not args.no_image_caption):
            res = pipe.ingest_batch(b)
            log.append({"segment": b.experiences[0].segment_id, "chars": len(b.experiences[0].text),
                        "sealed": len(res.sealed_event_ids), "unclosed": res.unclosed_count,
                        "errors": res.errors})
    # 对话结束：force_save 封存残留未完成事件，并补建索引
    last = sessions[-1]
    flush_mem = L.session_batches(sample, last, subject_id=subject_id, soft_chars=10**9)[0] \
        .experiences[0].model_copy(update={"text": "", "segment_id": f"{last.session_id}_flush"})
    flushed = pipe.metabolism_service.process_input("", force_save=True, memory=flush_mem)
    for ev in flushed:
        pipe.recall_pipeline.index_event(ev)
    return {"sample_id": sid, "sessions": len(sessions), "batches": len(log),
            "flush_sealed": len(flushed), "wall_s": round(time.perf_counter() - t0, 1), "log": log}


def event_sample(ev) -> str | None:
    """事件来源对话：input_id（未落库时为空）→ 归属段 segment_id → 对象 id 前缀 ``{sample}:``。"""
    if ev is None:
        return None
    sid = L.sample_of_segment(getattr(ev, "input_id", None))
    if sid:
        return sid
    for a in getattr(ev, "interlocutor_attributions", None) or []:
        sid = L.sample_of_segment(getattr(a, "segment_id", None))
        if sid:
            return sid
    for r in ev.role_list or []:
        if not r.is_subject and r.role_id and ":" in r.role_id:
            return r.role_id.split(":", 1)[0]
    return None


def eval_sample(pipe, qa_llm, sample, all_sessions, args, subject_id) -> list[dict]:
    sid = sample["sample_id"]
    conv = sample["conversation"]
    tol = timedelta(minutes=args.time_tolerance_min)
    qas = sample["qa"][: args.max_questions] if args.max_questions else sample["qa"]
    records = []
    for order, qa in enumerate(qas):
        frags = pipe.recall(subject_id, qa["question"], budget_chars=args.budget_chars,
                            limit=args.limit, reinforce=not args.no_reinforce)
        mappings = []
        for f in frags:
            if f.type == "unclosed":
                continue
            ev = pipe.event_repo.get(f.event_id)
            src_sample = event_sample(ev)
            occ = (ev.occurred_at if ev else None) or f.occurred_at
            src = (ev.source_ids if ev else None) or f.source_ids
            m = L.map_event_to_session(all_sessions, f.event_id, occ, src, tolerance=tol,
                                       sample_id=src_sample or sid,
                                       content_raw=(ev.content_raw if ev else None))
            m.cross_conversation = src_sample is not None and src_sample != sid
            mappings.append(m)
        same = [m for m in mappings if not m.cross_conversation]
        prompt = L.build_answer_prompt(args.answer_style, a=conv["speaker_a"], b=conv["speaker_b"],
                                       memories=fmt_memories(frags, args.context_chars),
                                       question=qa["question"])
        pred = qa_llm.complete("default", [{"role": "user", "content": prompt}], temperature=0.0).strip()
        gold = qa.get("answer", qa.get("adversarial_answer", ""))
        judge = None
        if args.llm_judge and qa["category"] != 5:
            v = qa_llm.complete("default", [{"role": "user", "content": JUDGE_PROMPT.format(
                q=qa["question"], gold=gold, pred=pred)}], temperature=0.0)
            judge = "CORRECT" in v.upper()
        records.append({
            "sample_id": sid, "order": order, "question": qa["question"],
            "category": qa["category"], "gold": gold, "pred": pred,
            "f1": L.score_answer(qa["category"], pred, gold), "judge": judge,
            "n_fragments": len(frags),
            "recalled": [m.__dict__ for m in mappings],
            "unmatched_events": sum(1 for m in mappings if not m.matched),
            # 证据命中只认本对话事件（跨对话事件 dia_id 命名空间不同）
            "evidence_hits": L.evidence_hits(same, qa.get("evidence", [])),
        })
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/LoCoMo/locomo10.json")
    ap.add_argument("--out", default="runs/locomo")
    ap.add_argument("--samples", nargs="*", default=None, help="默认只取第 1 段对话")
    ap.add_argument("--all", action="store_true", help="全部 10 段共用一个实例（需评审同意）")
    ap.add_argument("--subject", default=L.PROTAGONIST,
                    help="全局中立主体 id；每段对话映射到该对话 speaker_a")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--thinking", choices=["off", "default"], default="off")
    ap.add_argument("--soft-chars", type=int, default=3000)
    ap.add_argument("--context-chars", type=int, default=3000)
    ap.add_argument("--budget-chars", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--time-tolerance-min", type=float, default=5.0)
    ap.add_argument("--lexical-latin-mode", choices=["overlap", "token", "off"], default="overlap")
    ap.add_argument("--perspective", choices=["first_person", "neutral"], default="first_person",
                    help="neutral=中立视角+按原文语言写摘要（同一开关，默认关）")
    ap.add_argument("--answer-style", choices=list(L.ANSWER_STYLES), default="strict",
                    help="english_lenient=强制英文作答，仅在完全无证据时答 Not mentioned")
    ap.add_argument("--no-reinforce", action="store_true")
    ap.add_argument("--embedding-model", default=None,
                    help="e.g. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--embedding-version", type=int, default=None)
    ap.add_argument("--context-window", type=int, default=88000)
    ap.add_argument("--llm-judge", action="store_true")
    ap.add_argument("--no-image-caption", action="store_true")
    ap.add_argument("--max-questions", type=int, default=0)
    args = ap.parse_args(argv)
    if not os.environ.get("REMS_LLM__API_KEY"):
        sys.exit("REMS_LLM__API_KEY not set")
    with open(args.data, encoding="utf-8") as fh:
        data = json.load(fh)
    if args.all:
        chosen = data
    elif args.samples:
        chosen = [s for s in data if s["sample_id"] in set(args.samples)]
    else:
        chosen = data[:1]
    subject_id = args.subject
    out = Path(args.out)
    (out / "instance").mkdir(parents=True, exist_ok=True)
    params = dict(vars(args), subject_id=subject_id, samples_run=[s["sample_id"] for s in chosen],
                  subject_alias={s["sample_id"]: s["conversation"]["speaker_a"] for s in chosen})
    (out / "params.json").write_text(json.dumps(params, indent=2))
    cfg = build_config(args, out / "instance")
    pipe = REMSPipeline.from_config(cfg)  # 单实例，所有对话共用
    qa_llm = LLMProvider(cfg)
    all_sessions = [s for smp in data for s in L.iter_sessions(smp)]
    t = time.perf_counter()
    ingest = [ingest_sample(pipe, s, args, subject_id) for s in chosen]  # 先全部写入
    ingest_usage = _usage(pipe.llm.invocation_history())
    t1 = time.perf_counter()
    records = [r for s in chosen for r in eval_sample(pipe, qa_llm, s, all_sessions, args, subject_id)]
    summary = L.summarize(records)
    invs = pipe.llm.invocation_history() + qa_llm.invocation_history()
    summary.update({
        "params": params, "events": len(pipe.event_repo.list_all()),
        "ingest_wall_s": round(t1 - t, 1), "qa_wall_s": round(time.perf_counter() - t1, 1),
        "ingest_llm": ingest_usage, "qa_llm": _usage(qa_llm.invocation_history()),
        "response_models": sorted({i.response_model or "?" for i in invs}),
        "ingest": [{k: v for k, v in i.items() if k != "log"} for i in ingest],
    })
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records))
    (out / "ingest_log.json").write_text(json.dumps(ingest, indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
