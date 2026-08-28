"""修复 conversation.jsonl 中被编码破坏的 deepseek 长回复（R23/R24/R25/R27）。

替换规则：按 0-based 行索引定位，保留 speaker/ts/ingested_run，仅替换 content。
正确内容来自 fix_inputs/ 下 apply_patch 写入的 UTF-8 文件。
"""

from __future__ import annotations

import json
from pathlib import Path

_BASE = Path(__file__).resolve().parent
CONV = _BASE / "conversation.jsonl"
INPUTS = _BASE / "fix_inputs"

# 0-based 行号 → 正确内容文件
FIXES = {
    22: "r23.txt",  # deepseek 推荐方案
    23: "r24.txt",  # 事件/对象/感情与回忆关系
    24: "r25.txt",  # 更改方案
    26: "r27.txt",  # 记忆质量报告
}


def main() -> int:
    rows = [
        json.loads(l)
        for l in CONV.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 27, f"unexpected row count: {len(rows)}"
    for idx, name in FIXES.items():
        content = (INPUTS / name).read_text(encoding="utf-8").strip()
        row = rows[idx]
        assert row["speaker"] == "deepseek", f"R{idx+1} speaker mismatch"
        assert row["content"].count("?") > 20, f"R{idx+1} not garbled?"
        row["content"] = content
        print(f"R{idx+1} fixed: {len(content)} chars, ?={content.count('?')}")
    CONV.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in rows) + "\n",
        encoding="utf-8",
    )
    print("conversation.jsonl rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
