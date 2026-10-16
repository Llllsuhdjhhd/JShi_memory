"""Budgeted LoCoMo evaluation through the adjacent real JShi core.

Default: one complete history and 30 stratified questions; prepare/ingest/answer/
score/report phases can be resumed independently without repeating completed calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rems.config import REMSConfig, UserMode
from rems.eval.budget import BudgetLedger
from rems.eval.client import EvaluationLLM
from rems.eval.dataset import (CATEGORIES, canonical_answer, question_id, questions,
                               read_dataset, render_question, sessions)
from rems.eval.runtime import (SUBJECT, answer_question, close_pipeline, finalize_history,
                              load_jshi, replay_history)
from rems.eval.scoring import aggregate, build_atoms, coverage, judge, official_f1
from rems.pipeline import REMSPipeline


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def fingerprint_code(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def make_config(directory, api_key, *, embedding="BAAI/bge-small-en-v1.5", english_word_lexical=False):
    # Evaluation settings must not inherit unrelated daily-runtime knobs from .env.
    cfg = REMSConfig(_env_file=None, _env_prefix="JSHI_EVAL_UNUSED_")
    cfg.subject_id = SUBJECT
    cfg.recall_english_word_lexical = english_word_lexical
    cfg.llm.api_key = api_key
    cfg.llm.base_url = "https://api.deepseek.com"
    cfg.llm.max_retries = 0
    for field in type(cfg.llm.task_models).model_fields:
        setattr(cfg.llm.task_models, field, "deepseek-flash")
    cfg.embedding.provider = "local"
    cfg.embedding.model_name = embedding
    cfg.embedding.query_instruction = "Represent this sentence for searching relevant passages: "
    cfg.recall_embedding_version = 2
    cfg.storage.database_url = "sqlite:///" + (directory / "rems.db").resolve().as_posix()
    cfg.storage.qdrant_path = str((directory / "qdrant").resolve())
    cfg.storage.qdrant_collection = "locomo_english"
    cfg.user_mode = UserMode.MULTI
    return cfg


def make_pipe(directory, cfg, llm, encoder):
    config = cfg.model_copy(deep=True)
    config.storage.database_url = "sqlite:///" + (directory / "rems.db").resolve().as_posix()
    config.storage.qdrant_path = str((directory / "qdrant").resolve())
    pipe = REMSPipeline.from_config(config, llm=llm)
    pipe.recall_pipeline._embedding._model = encoder
    return pipe


def report(out, manifest, ledger):
    records = []
    for item in manifest["questions"]:
        path = out / "grades" / (item["safe_id"] + ".json")
        if path.exists():
            records.append(read_json(path))
        else:
            records.append({"status": "pending", "category": item["category"]})
    summary = aggregate(records)
    summary["pending"] = sum(r["status"] == "pending" for r in records)
    summary["errors"] = sum(r["status"] == "error" for r in records)
    summary["by_category"] = {CATEGORIES[c]: aggregate([r for r in records if r["category"] == c])
                               for c in sorted({r["category"] for r in records})}
    summary["budget"] = ledger.summary()
    summary["scope"] = manifest["scope"]
    provenance = out / "baseline_provenance.json"
    if provenance.exists():
        summary["baseline_provenance"] = read_json(provenance)
    comparisons = []
    for item in manifest["questions"]:
        cid = item["safe_id"] + ".json"
        if (out / "control_grades" / cid).exists() and (out / "grades" / cid).exists():
            main_grade = read_json(out / "grades" / cid)
            control_grade = read_json(out / "control_grades" / cid)
            if main_grade.get("status") != "scored" or control_grade.get("status") != "scored":
                continue
            comparisons.append({"question_id": item["question_id"],
                "with_recall": main_grade["grade"]["answer_verdict"],
                "without_recall": control_grade["grade"]["answer_verdict"]})
    summary["controls"] = comparisons
    write_json(out / "summary.json", summary)
    def percent(value):
        return "不适用" if value is None else f"{value:.1%}"
    lines = ["# 匠石 LoCoMo 评测结果", "", f"已评分 {summary['scored']} / {summary['questions']} 题；待处理 {summary['pending']} 题；错误 {summary['errors']} 题。", "",
        "| 指标 | 结果 |", "|---|---|",
        f"| 官方 F1（按题型） | {percent(summary['official_f1'])} |",
        f"| 最终回答语义正确率 | {percent(summary['semantic_accuracy'])} |",
        f"| 直接答案条目覆盖率 | {percent(summary['direct'])} |",
        f"| 充分信息覆盖率 | {percent(summary['sufficient'])} |",
        f"| 完整覆盖题目比例 | {percent(summary['complete'])} |",
        f"| 冲突题目比例 | {percent(summary['conflict'])} |", "",
        "以上为自动评分，尚未完成人工抽查，语义评分不是绝对真值。", "",
        "## 分题型", "", "| 题型 | 已评分 | 官方 F1 | 语义正确率 | 充分信息覆盖 |", "|---|---:|---:|---:|---:|"]
    for name, row in summary["by_category"].items():
        lines.append(f"| {name} | {row['scored']} | {percent(row['official_f1'])} | {percent(row['semantic_accuracy'])} | {percent(row['sufficient'])} |")
    budget = summary["budget"]
    lines += ["", "## 成本与范围", "",
        f"全评测累计保守计费估算：¥{budget['known_cost_yuan']:.4f}；未结算预留：¥{budget['held_yuan']:.4f}。",
        "按 Flash 高峰价格、输入不享缓存优惠估算；这不是供应商实际扣款金额。", "",
        "使用真实 JShi 默认木头人格文本入口、CognitionSkill、材料组装、MemoryShell、REMS3 适配与记忆后端。",
        "历史固定回放，经匠石经历台账和记忆控制写入，不生成替代历史回复；结束时强制封存残留。",
        "不覆盖语音/JEV 路由、自主多轮回忆、后台长期经验维护。测试使用英文及观察者视角叠加指令。",
        "每题复制同一封存记忆库，并新建核心现场；只给评分程序提供答案。",
        "小库结果不能证明已验证大库概率采样、长期遗忘和持续学习。", "",
        "测试时间锚点设为历史最后一个会话次日；文本入口不经过语音 JEV。", "",
        "## 关闭回忆对照", ""]
    for row in comparisons:
        lines.append(f"- {row['question_id']}：有回忆 {row['with_recall']}；无回忆 {row['without_recall']}。")
    lines += ["",
        "## 逐题明细", ""]
    for item in manifest["questions"]:
        qid = item["safe_id"]
        lines.append(f"- {item['question_id']}：{item['question']} — [评分](grades/{qid}.json)，[实际核心输入与回答](answers/{qid}.json)")
    (out / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT.parent / "Locomo-Plus/data/locomo10.json")
    ap.add_argument("--jshi", type=Path, default=ROOT.parent / "JShi")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/locomo_smoke")
    ap.add_argument("--phase", choices=["prepare", "ingest", "finalize", "answer", "score", "report", "run"], default="run")
    ap.add_argument("--conversations", type=int, default=1)
    ap.add_argument("--questions", type=int, default=30, help="Stratified questions per conversation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--english-word-lexical", action="store_true", help="Use whole-word English lexical coverage")
    ap.add_argument("--default-memory-triggers", action="store_true", help="Use real JShi default trigger conditions with synchronous deterministic dispatch")
    ap.add_argument("--event-index-level", choices=["L0", "L1"], default="L0")
    ap.add_argument("--reuse-baseline", type=Path, help="Reuse sealed events and frozen rubrics; rebuild only the index")
    ap.add_argument("--budget", type=float, default=10, help="Cumulative submission limit, maximum 90 yuan")
    ap.add_argument("--no-recall-controls", type=int, default=3)
    args = ap.parse_args(argv)
    if args.conversations < 1 or args.questions < 1 or args.no_recall_controls < 0:
        ap.error("Invalid sample counts")
    # Keep all artifacts and credentials separate. A report never constructs an API client.
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(ROOT / "runs/locomo_budget.jsonl", args.budget)
    manifest_path = out / "manifest.json"
    if args.phase == "report":
        print(json.dumps(report(out, read_json(manifest_path), ledger), ensure_ascii=False, indent=2))
        return
    data, data_hash = read_dataset(args.data)
    if args.conversations > len(data):
        ap.error("Requested more conversations than available")
    load_jshi(args.jshi)
    cfg = make_config(out / "unused", "not-loaded", english_word_lexical=args.english_word_lexical)
    cfg.recall_event_index_level = args.event_index_level
    source_hash = fingerprint_code(ROOT / "src/rems")
    jshi_hash = fingerprint_code(args.jshi / "src/jshi")
    selected = data[:args.conversations]
    selection = [{"sample_id": s["sample_id"], "index": i, "question_id": question_id(s, i),
                  "safe_id": f"c{n}_q{i}", "category": int(qa["category"]),
                  "question": qa["question"],
                  "input_question": render_question(qa, args.seed+n+i)[0],
                  "answer_options": render_question(qa, args.seed+n+i)[1],
                  "reference": "Not mentioned in the conversation" if int(qa["category"]) == 5 else str(qa.get("answer", "")),
                  "evidence": qa.get("evidence", [])}
                 for n, s in enumerate(selected) for i, qa in questions(s, args.questions, args.seed+n)]
    settings = {"dataset_sha256": data_hash, "rems_code_sha256": source_hash,
        "jshi_code_sha256": jshi_hash, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "conversations": args.conversations, "questions_per_conversation": args.questions,
        "default_memory_triggers": args.default_memory_triggers,
        "event_index_level": args.event_index_level,
        "reuse_baseline": str(args.reuse_baseline.resolve()) if args.reuse_baseline else None,
        "no_recall_controls": args.no_recall_controls, "embedding": "BAAI/bge-small-en-v1.5",
        "category5_protocol": "official_distractor_vs_not_mentioned_choices",
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "scope": "real JShi wood text-entry core; teacher-forced ledger ingestion; event-memory baseline",
        "evaluation_config": cfg.model_dump(mode="json", exclude={
            "llm": {"api_key"}, "embedding": {"api_key"}, "storage": True})}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        for key, value in settings.items():
            if manifest.get(key) != value:
                raise RuntimeError(f"Frozen run differs in {key}; use a new output directory")
    else:
        manifest = {**settings, "questions": selection}
        write_json(manifest_path, manifest)

    source = args.reuse_baseline.resolve() if args.reuse_baseline else None
    if source:
        previous = read_json(source / "manifest.json")
        for key in ("dataset_sha256", "jshi_code_sha256", "seed", "questions"):
            if previous[key] != manifest[key]:
                raise RuntimeError(f"Cannot reuse unpaired baseline: {key}")
        if not (out / "rubrics").exists():
            shutil.copytree(source / "rubrics", out / "rubrics")

    from dotenv import dotenv_values
    credentials = dotenv_values(ROOT / ".env")
    api_key = os.environ.get("REMS_LLM__API_KEY") or credentials.get("REMS_LLM__API_KEY")
    if not api_key:
        raise RuntimeError("REMS_LLM__API_KEY is required")
    cfg.llm.api_key = api_key
    llm = EvaluationLLM(cfg, ledger, out / "api_audit.jsonl")
    run_all = args.phase == "run"
    try:
        if args.phase == "prepare" or run_all:
            for item in manifest["questions"]:
                rubric = out / "rubrics" / (item["safe_id"] + ".json")
                if not rubric.exists():
                    atoms = [] if item["category"] == 5 else build_atoms(llm, item["question"], item["reference"])
                    write_json(rubric, {"question_id": item["question_id"], "atoms": atoms})
                    print("rubric " + item["question_id"], flush=True)
            rubric_hash = hashlib.sha256(json.dumps([
                read_json(out / "rubrics" / (q["safe_id"] + ".json")) for q in manifest["questions"]
            ], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if manifest.get("rubrics_sha256") not in (None, rubric_hash):
                raise RuntimeError("Previously frozen rubric changed; refuse to silently refreeze")
            manifest["rubrics_sha256"] = rubric_hash
            write_json(manifest_path, manifest)

        encoder = None
        if args.phase in {"ingest", "finalize", "answer"} or run_all:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(cfg.embedding.model_name, local_files_only=True, device="cpu")
            if encoder.get_sentence_embedding_dimension() != 384:
                raise RuntimeError("Unexpected English encoder dimension")

        if args.phase in {"ingest", "finalize"} or run_all:
            for n, sample in enumerate(selected):
                baseline = out / "baselines" / f"c{n}"
                if (baseline / "complete.json").exists():
                    continue
                if source and not baseline.exists():
                    original = source / "baselines" / f"c{n}"
                    info = read_json(original / "complete.json")
                    shutil.copytree(original, baseline)
                    # Completion is published only after the new index is fully built.
                    (baseline / "complete.json").unlink()
                    pipe = make_pipe(baseline, cfg, llm, encoder)
                    try:
                        scanned, written = pipe.recall_pipeline.backfill_index()
                    finally:
                        close_pipeline(pipe)
                    source_sha = hashlib.sha256((original / "rems.db").read_bytes()).hexdigest()
                    target_sha = hashlib.sha256((baseline / "rems.db").read_bytes()).hexdigest()
                    if source_sha != target_sha:
                        raise RuntimeError("Reindex changed the sealed event database")
                    write_json(out / "baseline_provenance.json", {
                        "source": str(original), "source_database_sha256": source_sha,
                        "reindexed_database_sha256": target_sha, "events_scanned": scanned,
                        "points_written": written, "event_index_level": args.event_index_level,
                        "reingested": False})
                    write_json(baseline / "complete.json", info)
                    print(f"reindexed {scanned} events, {written} points", flush=True)
                    continue
                if (baseline / "rems.db").exists() and args.phase != "finalize":
                    raise RuntimeError("Incomplete baseline retained; use a new run directory (no duplicate ingestion)")
                baseline.mkdir(parents=True, exist_ok=True)
                pipe = make_pipe(baseline, cfg, llm, encoder)
                try:
                    if args.phase == "finalize":
                        checkpoint = read_json(out / "history" / f"c{n}" / "history_complete.json")
                        if checkpoint["sample_id"] != sample["sample_id"]:
                            raise RuntimeError("History checkpoint belongs to another sample")
                        result = finalize_history(pipe, checkpoint)
                    else:
                        result = replay_history(pipe, llm, out / "history" / f"c{n}", sample,
                            default_triggers=args.default_memory_triggers)
                    pipe.db.engine.dispose()
                finally:
                    close_pipeline(pipe)
                write_json(baseline / "complete.json", result)

        if args.phase == "answer" or run_all:
            for order, item in enumerate(manifest["questions"]):
                n = int(item["safe_id"].split("_")[0][1:])
                baseline = out / "baselines" / f"c{n}"
                info = read_json(baseline / "complete.json")
                from datetime import datetime
                date = datetime.fromisoformat(info["last_date"]) + timedelta(days=1)
                variants = [("answers", "question_states", True)]
                if order < args.no_recall_controls:
                    variants.append(("control_answers", "controls", False))
                for answer_dir, work_dir, recall in variants:
                    dest = out / answer_dir / (item["safe_id"] + ".json")
                    if dest.exists():
                        continue
                    # No answer rubric/reference is passed across this boundary.
                    work = out / work_dir / item["safe_id"]
                    if work.exists():
                        raise RuntimeError("Incomplete question state retained; inspect before retrying")
                    shutil.copytree(baseline, work / "memory")
                    pipe = make_pipe(work / "memory", cfg, llm, encoder)
                    try:
                        answer = answer_question(pipe, llm, work / "core", info["names"],
                            item["input_question"], date, args.seed+order, recall=recall)
                    finally:
                        close_pipeline(pipe)
                    write_json(dest, {"question_id": item["question_id"], **answer})
                    print(answer_dir + " " + item["question_id"], flush=True)

        if args.phase == "score" or run_all:
            if not manifest.get("rubrics_sha256"):
                raise RuntimeError("Prepare and freeze all rubrics before scoring")
            current_hash = hashlib.sha256(json.dumps([
                read_json(out / "rubrics" / (q["safe_id"] + ".json")) for q in manifest["questions"]
            ], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if current_hash != manifest["rubrics_sha256"]:
                raise RuntimeError("Frozen rubrics changed; refuse scoring")
            for item in manifest["questions"]:
                rubric = read_json(out / "rubrics" / (item["safe_id"] + ".json"))
                for answer_dir, grade_dir in [("answers", "grades"), ("control_answers", "control_grades")]:
                    answer_path = out / answer_dir / (item["safe_id"] + ".json")
                    dest = out / grade_dir / (item["safe_id"] + ".json")
                    if dest.exists() or not answer_path.exists():
                        continue
                    answer = read_json(answer_path)
                    try:
                        predicted = canonical_answer(answer["answer"], item["answer_options"])
                        grade = judge(llm, question=item["input_question"], reference=item["reference"],
                            atoms=rubric["atoms"], memory=answer["delivered_memory"], answer=predicted,
                            adversarial=item["category"] == 5, as_of=answer.get("as_of"))
                    except ValueError as exc:
                        write_json(dest, {"question_id": item["question_id"], "category": item["category"],
                                          "status": "error", "error": str(exc)})
                        print("invalid judge " + item["question_id"], flush=True)
                        continue
                    record = {"question_id": item["question_id"], "category": item["category"], "status": "scored",
                              "grade": grade, "coverage": coverage(grade),
                              "canonical_answer": predicted,
                              "official_f1": official_f1(item["category"], predicted, item["reference"])}
                    write_json(dest, record)
                    print(grade_dir + " " + item["question_id"], flush=True)
    finally:
        summary = report(out, manifest, ledger)
        print(json.dumps({"scored": summary["scored"], "budget": summary["budget"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
