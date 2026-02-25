#!/usr/bin/env python3
"""Smoke test: call every configured model once and print response text.

Usage:
  python test_models_smoke.py
  python test_models_smoke.py --prompt "Write a 3-line story about rain."
  python test_models_smoke.py --prompt-id 28 --modifier-index 0
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pipeline import (
    DATASET_PATH,
    MODEL_SPECS,
    HttpClient,
    ModelClient,
    build_final_prompt,
    load_prompt_items,
)

MODEL_SPECS = [
    {"display_name": "gemini-3-flash-preview", "provider": "gemini", "api_model": "gemini-3-flash-preview", "thinking": "non_reasoning"},
    {"display_name": "gemini-3-pro-preview", "provider": "gemini", "api_model": "gemini-3-pro-preview", "thinking": "reasoning"},
    {"display_name": "GPT-5.2 pro (reasoning low)", "provider": "openai", "api_model": "gpt-5.2", "thinking": "non_reasoning"},
    {"display_name": "GPT-5.2 pro (reasoning high)", "provider": "openai", "api_model": "gpt-5.2", "thinking": "reasoning"},
    {"display_name": "claude-opus-4-6 (thinking disabled)", "provider": "anthropic", "api_model": "claude-opus-4-6", "thinking": "non_reasoning"},
    {"display_name": "claude-opus-4-6 (thinking enabled)", "provider": "anthropic", "api_model": "claude-opus-4-6", "thinking": "reasoning"},
    {"display_name": "grok-4-1-fast-non-reasoning", "provider": "xai", "api_model": "grok-4-1-fast-non-reasoning", "thinking": "non_reasoning"},
    {"display_name": "grok-4-1-fast-reasoning", "provider": "xai", "api_model": "grok-4-1-fast-reasoning", "thinking": "reasoning"},
]



def build_test_prompt(
    prompt: Optional[str],
    prompt_id: Optional[int],
    modifier_index: int,
    dataset_path: Path,
) -> str:
    if prompt:
        return prompt

    if prompt_id is not None:
        rows, missing = load_prompt_items(dataset_path, [prompt_id], max_modifiers=None)
        if missing or not rows:
            raise ValueError(f"prompt_id={prompt_id} not found in {dataset_path}")
        row = rows[0]
        mods = row.get("seed_modifiers", [])
        if not mods:
            return row["writing_prompt"]
        if modifier_index < 0 or modifier_index >= len(mods):
            raise ValueError(
                f"modifier_index={modifier_index} out of range for prompt_id={prompt_id}; "
                f"valid: 0..{len(mods)-1}"
            )
        return build_final_prompt(row["writing_prompt"], mods[modifier_index])

    # Default compact test prompt for quick smoke testing.
    return (
        "Write a short creative paragraph (80-120 words) about a city where every clock "
        "runs at a slightly different speed. Use vivid sensory detail."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test all configured generation models")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt text")
    parser.add_argument("--prompt-id", type=int, default=None, help="Load prompt by id from creative_writing_prompts_v3.json")
    parser.add_argument("--modifier-index", type=int, default=0, help="Seed modifier index when using --prompt-id")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--out-json",
        type=str,
        default="outputs/smoke_test_results.json",
        help="Path to save smoke test JSON results",
    )
    args = parser.parse_args()

    load_dotenv()

    test_prompt = build_test_prompt(
        prompt=args.prompt,
        prompt_id=args.prompt_id,
        modifier_index=args.modifier_index,
        dataset_path=DATASET_PATH,
    )

    http = HttpClient(timeout_seconds=args.timeout, max_retries=args.retries)
    client = ModelClient(http)

    print("=" * 80)
    print("SMOKE TEST PROMPT")
    print("=" * 80)
    print(test_prompt)
    print()

    results = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": test_prompt,
        "config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "retries": args.retries,
        },
        "models": [],
    }

    for spec in MODEL_SPECS:
        display_name = spec["display_name"]
        print("=" * 80)
        print(f"MODEL: {display_name}")
        print("=" * 80)
        started = time.time()
        model_result = {
            "display_name": display_name,
            "provider": spec.get("provider"),
            "api_model": spec.get("api_model"),
            "thinking": spec.get("thinking"),
            "latency_sec": None,
            "response_text": None,
            "error": None,
        }
        try:
            text, _raw = client.generate(
                model_spec=spec,
                prompt=test_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            model_result["response_text"] = text
            if text:
                print(text)
            else:
                print("[empty response text]")
        except Exception as e:
            model_result["error"] = str(e)
            print(f"[ERROR] {e}")
        model_result["latency_sec"] = round(time.time() - started, 3)
        results["models"].append(model_result)
        print()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
