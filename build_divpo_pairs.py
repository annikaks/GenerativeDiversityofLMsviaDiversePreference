#!/usr/bin/env python3
"""Build DivPO preference pairs for Qwen3-8B without training.

This is the data-construction half of DivPO:
- prompts 0..26 only
- base prompt + all 10 seed modifiers
- batched candidate generation
- embedding-based quality proxy + marginal diversity contribution
- pairwise preference construction

Designed to run locally or on Modal. Outputs JSONL files consumed by
`train_divpo_qwen_tinker.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from train_divpo_qwen import (
    MODEL_ID,
    THINKING_MODES,
    build_preference_dataset,
    load_qwen_model,
    slugify,
)


DEFAULT_OUTPUT_ROOT = Path("outputs/divpo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DivPO preference pairs for Qwen3-8B")
    parser.add_argument(
        "--thinking-mode",
        choices=["non_reasoning", "reasoning", "both"],
        default="both",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--quality-floor-quantile", type=float, default=0.4)
    parser.add_argument("--lambda-quality", type=float, default=1.0)
    parser.add_argument("--lambda-diversity", type=float, default=1.5)
    parser.add_argument("--subset-size", type=int, default=4)
    parser.add_argument("--negatives-per-step", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    args.output_root.mkdir(parents=True, exist_ok=True)
    modes = THINKING_MODES if args.thinking_mode == "both" else (args.thinking_mode,)

    for thinking_mode in modes:
        mode_slug = slugify(f"qwen3-8b-{thinking_mode}-divpo")
        output_dir = args.output_root / mode_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"[divpo-build] loading model={MODEL_ID} mode={thinking_mode}")
        tokenizer, model = load_qwen_model(MODEL_ID)
        build_preference_dataset(
            thinking_mode=thinking_mode,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            samples_per_prompt=args.samples_per_prompt,
            generation_batch_size=args.generation_batch_size,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            embedding_batch_size=args.embedding_batch_size,
            quality_floor_quantile=args.quality_floor_quantile,
            lambda_quality=args.lambda_quality,
            lambda_diversity=args.lambda_diversity,
            subset_size=args.subset_size,
            negatives_per_step=args.negatives_per_step,
        )
        print(f"[divpo-build] saved pairs to {output_dir}")


if __name__ == "__main__":
    main()
