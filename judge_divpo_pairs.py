#!/usr/bin/env python3
"""Annotate existing DivPO pairs with LLM-as-judge accuracy.

This script does not regenerate pairs. It:
1. Reads an existing `preference_pairs.jsonl`
2. Deduplicates unique (prompt, response) items across chosen/rejected fields
3. Scores each unique response once with the existing judge pipeline logic
4. Adds `chosen_llm_accuracy` / `rejected_llm_accuracy` to each pair

Outputs:
- annotated pair file
- response-level cache
- summary JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from run_accuracy_batched import HttpClient, RateLimiter, call_judge


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(prompt: str, response: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"\n<SEP>\n")
    h.update(response.encode("utf-8"))
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge existing DivPO pairs and annotate with LLM accuracy")
    parser.add_argument("--input", type=Path, required=True, help="Existing preference_pairs.jsonl")
    parser.add_argument("--annotated-output", type=Path, required=True, help="Annotated output JSONL path")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional summary JSON path")
    parser.add_argument("--judge-provider", type=str, default="anthropic", choices=["anthropic", "gemini", "xai"])
    parser.add_argument("--judge-model", type=str, default="claude-opus-4-6")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    load_dotenv()
    rows = load_jsonl(args.input)

    unique_items: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for side in ("chosen", "rejected"):
            prompt = row["prompt"]
            response = row[side]
            key = stable_key(prompt, response)
            if key not in unique_items:
                unique_items[key] = {
                    "prompt": prompt,
                    "response": response,
                    "score": None,
                    "error": None,
                }

    http = HttpClient(timeout_seconds=args.timeout, max_retries=args.retries)
    limiter = RateLimiter(requests_per_minute=args.rpm)

    response_cache_path = args.annotated_output.with_suffix(".response_cache.json")
    cache_doc: Dict[str, Any]
    if response_cache_path.exists():
        with response_cache_path.open("r", encoding="utf-8") as f:
            cache_doc = json.load(f)
    else:
        cache_doc = {
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "judge_config": {
                "provider": args.judge_provider,
                "model": args.judge_model,
                "temperature": args.judge_temperature,
                "rpm": args.rpm,
            },
            "entries": {},
        }

    total_unique = len(unique_items)
    started = time.time()
    for idx, (key, item) in enumerate(unique_items.items(), start=1):
        cached = cache_doc["entries"].get(key)
        if cached and cached.get("score") is not None:
            item["score"] = cached["score"]
            item["error"] = cached.get("error")
            continue

        try:
            parsed = call_judge(
                judge_provider=args.judge_provider,
                http=http,
                limiter=limiter,
                judge_model=args.judge_model,
                judge_temperature=args.judge_temperature,
                prompt_text=item["prompt"],
                responses=[
                    {
                        "sample_index": 0,
                        "record_id": key,
                        "response_text": item["response"],
                    }
                ],
            )
            score_items = parsed.get("scores", [])
            if not score_items:
                raise RuntimeError("Judge returned empty scores list")
            score = int(score_items[0]["score"])
            item["score"] = score
            item["error"] = None
        except Exception as e:
            item["score"] = None
            item["error"] = str(e)

        cache_doc["entries"][key] = {
            "score": item["score"],
            "error": item["error"],
        }
        save_json(response_cache_path, cache_doc)
        elapsed_min = (time.time() - started) / 60.0
        remaining = total_unique - idx
        eta_min = remaining / args.rpm if args.rpm > 0 else 0.0
        print(
            f"[judge-pairs] unique={idx}/{total_unique} score={item['score']} "
            f"elapsed_min={elapsed_min:.1f} eta_min~{eta_min:.1f}"
        )

    annotated_rows: List[Dict[str, Any]] = []
    num_missing_scores = 0
    for row in rows:
        chosen_key = stable_key(row["prompt"], row["chosen"])
        rejected_key = stable_key(row["prompt"], row["rejected"])
        chosen_score = unique_items[chosen_key]["score"]
        rejected_score = unique_items[rejected_key]["score"]

        annotated = dict(row)
        annotated["chosen_llm_accuracy"] = chosen_score
        annotated["rejected_llm_accuracy"] = rejected_score
        annotated_rows.append(annotated)

        if chosen_score is None or rejected_score is None:
            num_missing_scores += 1

    save_jsonl(args.annotated_output, annotated_rows)
    summary_path = args.summary_out or args.annotated_output.with_suffix(".summary.json")
    save_json(
        summary_path,
        {
            "input_file": str(args.input),
            "annotated_output": str(args.annotated_output),
            "input_rows": len(rows),
            "annotated_rows": len(annotated_rows),
            "num_missing_scores": num_missing_scores,
            "num_unique_prompt_response_items": total_unique,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
        },
    )
    print(f"[saved] {args.annotated_output}")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
