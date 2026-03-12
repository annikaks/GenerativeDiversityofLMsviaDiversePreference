#!/usr/bin/env python3
"""Filter DivPO pair files down to rows with complete LLM accuracy annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep only DivPO pairs with complete LLM accuracy annotations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument(
        "--min-llm-accuracy",
        type=float,
        default=None,
        help="Optional minimum threshold applied to both chosen and rejected LLM accuracy",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    kept: List[Dict[str, Any]] = []
    dropped_missing = 0
    dropped_threshold = 0

    for row in rows:
        chosen_score = row.get("chosen_llm_accuracy")
        rejected_score = row.get("rejected_llm_accuracy")
        if chosen_score is None or rejected_score is None:
            dropped_missing += 1
            continue
        if args.min_llm_accuracy is not None and min(chosen_score, rejected_score) < args.min_llm_accuracy:
            dropped_threshold += 1
            continue
        kept.append(row)

    save_jsonl(args.output, kept)
    summary_path = args.summary_out or args.output.with_suffix(".summary.json")
    save_json(
        summary_path,
        {
            "input_file": str(args.input),
            "output_file": str(args.output),
            "input_rows": len(rows),
            "output_rows": len(kept),
            "dropped_missing_llm_accuracy": dropped_missing,
            "dropped_below_threshold": dropped_threshold,
            "min_llm_accuracy": args.min_llm_accuracy,
        },
    )
    print(f"[filter-llm] input_rows={len(rows)} output_rows={len(kept)}")
    print(
        f"[filter-llm] dropped_missing_llm_accuracy={dropped_missing} "
        f"dropped_below_threshold={dropped_threshold}"
    )
    print(f"[saved] {args.output}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
