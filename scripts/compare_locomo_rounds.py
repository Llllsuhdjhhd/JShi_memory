"""Produce auditable, paired results and actual recall samples from completed runs."""
import json
import argparse
from pathlib import Path

from locomo_jshi_eval import ROOT, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, default=ROOT / "runs/locomo_round1")
    parser.add_argument("--second", type=Path, default=ROOT / "runs/locomo_round2_clean")
    args = parser.parse_args()
    first, second = args.first, args.second
    manifests = [read_json(p / "manifest.json") for p in (first, second)]
    for key in ("dataset_sha256", "jshi_code_sha256", "seed", "questions", "rubrics_sha256"):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f"Unpaired comparison: {key}")
    summaries = [read_json(p / "summary.json") for p in (first, second)]
    if any(s["scored"] != 30 or s["errors"] or s["pending"] for s in summaries):
        raise ValueError("Both rounds must be complete before comparison")
    pairs = []
    for q in manifests[0]["questions"]:
        grades = [read_json(p / "grades" / (q["safe_id"] + ".json")) for p in (first, second)]
        pairs.append({"question_id": q["question_id"], "question": q["question"],
            "before": grades[0], "after": grades[1]})
    samples = []
    for index in (19, 130, 102, 12, 24, 47, 116, 16):
        q = next(q for q in manifests[0]["questions"] if q["index"] == index)
        samples.append({**q, "rounds": [
            {"answer": read_json(p / "answers" / (q["safe_id"] + ".json")),
             "grade": read_json(p / "grades" / (q["safe_id"] + ".json"))}
            for p in (first, second)]})
    provenance = read_json(second / "baseline_provenance.json") if (second / "baseline_provenance.json").exists() else {}
    source_run = Path(provenance["source"]).parents[1].resolve() if provenance.get("source") else None
    reused = (source_run == first.resolve() and provenance.get("reingested") is False
              and not provenance.get("additional_age_refinement")
              and provenance.get("view") != "small_events_only")
    result = {"summaries": summaries, "paired_questions": pairs, "samples": samples,
        "limitations": "Same development questions; stochastic model answers and grades, not independent human adjudication.",
        "same_sealed_memory_reindexed": reused}
    write_json(second / "comparison.json", result)
    lineage = ("复用同一封存事件数据库，仅重建索引；未重新提取或压缩事件。" if reused else
               "事件形成或后续提炼有变化，并非固定封存库的单变量对照；具体来源见各轮 manifest 与 baseline_provenance。")
    lines = ["# 两轮逐题自动结果与实际回忆", "", "同一完整历史、同一 30 题及固定评分条目；" + lineage, "",
        "| 指标 | 第一轮 | 第二轮 |", "|---|---:|---:|"]
    for key, title in (("official_f1", "官方 F1"), ("semantic_accuracy", "最终回答语义正确率"),
                       ("direct", "直接条目覆盖"), ("sufficient", "充分信息覆盖"),
                       ("complete", "完整覆盖题目比例")):
        lines.append(f"| {title} | {summaries[0][key]:.2%} | {summaries[1][key]:.2%} |")
    lines += ["", "自动语义评分并非人工裁决；同批开发题的变化不能视为独立泛化成绩。", "",
        "## 全部题目的配对结果", "", "| 题目 | 第一轮覆盖/回答 | 第二轮覆盖/回答 |", "|---|---|---|"]
    if (second / "reviewed_summary.json").exists():
        reviewed = read_json(second / "reviewed_summary.json")
        lines[2:2] = [f"自动评分有阅读复核修正。复核后充分信息覆盖率 {reviewed['sufficient']:.2%}，完整覆盖 {reviewed['complete']:.2%}，冲突 {reviewed['conflict']:.2%}。", "",
            "参见 [复核修正](reading_review.json)、[复核汇总](reviewed_summary.json)；下表及逐题原文保留自动原始结果。", ""]
    for pair in pairs:
        cells = []
        for key in ("before", "after"):
            grade = pair[key]
            cov = grade["coverage"]["sufficient"]
            cells.append(("不适用" if cov is None else f"{cov:.0%}") + "/" + grade["grade"]["answer_verdict"])
        lines.append(f"| {pair['question_id']} | {cells[0]} | {cells[1]} |")
    for sample in samples:
        lines += ["", "## " + sample["question_id"] + "：" + sample["question"], "",
            "参考答案：" + str(sample["reference"])]
        for i, round_ in enumerate(sample["rounds"], 1):
            lines += ["", f"### {(first, second)[i-1].name}", "", "最终回答：" + round_["answer"]["answer"], "",
                "实际交付回忆：", "", "```text", round_["answer"]["delivered_memory"], "```", "",
                "条目评分：", "", "```json", json.dumps(round_["grade"], ensure_ascii=False, indent=2), "```"]
    (second / "comparison.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(json.dumps({"before": summaries[0], "after": summaries[1]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
