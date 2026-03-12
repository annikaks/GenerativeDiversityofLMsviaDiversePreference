#!/usr/bin/env python3
"""Generate held-out prompts from a saved Tinker checkpoint.

Writes the same JSON schema used by the baseline generation pipeline so the
embedding and judge scripts can be reused unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from transformers import AutoTokenizer

import tinker
from tinker import types

from pipeline import DATASET_PATH, PROMPT_IDS_REQUESTED, build_final_prompt, load_prompt_items, slugify


MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_DIR = Path("outputs/generations")
DEFAULT_PROMPT_IDS = [28, 29, 30, 31, 32, 33]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_latest_checkpoint(checkpoints_path: Path) -> Dict[str, Any]:
    doc = json.load(checkpoints_path.open("r", encoding="utf-8"))
    checkpoints = doc.get("checkpoints", [])
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoints_path}")
    checkpoints = sorted(checkpoints, key=lambda row: int(row["step"]))
    return checkpoints[-1]


def make_safe_label(*parts: str) -> str:
    raw = "-".join(parts)
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe)


def build_prompt_text(tokenizer: Any, thinking_mode: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if "Qwen3" in MODEL_ID:
            kwargs["enable_thinking"] = thinking_mode == "reasoning"
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate held-out prompts from a Tinker checkpoint")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoints-json", type=Path, default=None)
    parser.add_argument("--thinking-mode", choices=["non_reasoning", "reasoning"], required=True)
    parser.add_argument("--alias", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=1100)
    parser.add_argument("--max-modifiers", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()

    checkpoint_path = args.checkpoint_path
    checkpoint_meta: Dict[str, Any] | None = None
    if checkpoint_path is None:
        if args.checkpoints_json is None:
            raise ValueError("Provide either --checkpoint-path or --checkpoints-json")
        checkpoint_meta = load_latest_checkpoint(args.checkpoints_json)
        checkpoint_path = checkpoint_meta["checkpoint_path"]

    alias = args.alias or slugify(
        f"posttrain-qwen3-8b-{args.thinking_mode}-{Path(checkpoint_path).name}"
    )
    out_path = args.output_dir / f"{alias}.json"

    prompt_items, missing = load_prompt_items(DATASET_PATH, DEFAULT_PROMPT_IDS, args.max_modifiers)
    if missing:
        print(f"[tinker-generate] missing_prompt_ids={missing}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    service_client = tinker.ServiceClient()
    sampler_checkpoint_path = checkpoint_path
    if "/weights/" in checkpoint_path:
        print(
            "[tinker-generate] training checkpoint provided; exporting sampler checkpoint first"
        )
        training_client = service_client.create_training_client_from_state(checkpoint_path)
        sampler_checkpoint_name = make_safe_label(alias, "sampler-export")
        sampler_checkpoint_path = training_client.save_weights_for_sampler(
            sampler_checkpoint_name
        ).result().path
        print(
            f"[tinker-generate] sampler_checkpoint={sampler_checkpoint_path}"
        )
    sampler = service_client.create_sampling_client(model_path=sampler_checkpoint_path)

    doc: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "source": "tinker-checkpoint",
        "checkpoint_path": checkpoint_path,
        "sampler_checkpoint_path": sampler_checkpoint_path,
        "checkpoint_meta": checkpoint_meta,
        "model": {
            "display_name": alias,
            "provider": "tinker",
            "api_model": MODEL_ID,
            "thinking": args.thinking_mode,
        },
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "num_samples": args.num_samples,
        },
        "records": [],
    }

    total_groups = len(prompt_items) * (1 + args.max_modifiers)
    group_index = 0
    for prompt_item in prompt_items:
        variants: List[Dict[str, Any]] = [
            {
                "seed_modifier_index": -1,
                "seed_modifier": "",
                "variant_type": "base_prompt",
                "final_prompt": prompt_item["writing_prompt"],
            }
        ]
        for idx, modifier in enumerate(prompt_item["seed_modifiers"][: args.max_modifiers]):
            variants.append(
                {
                    "seed_modifier_index": idx,
                    "seed_modifier": modifier,
                    "variant_type": "seed_modified_prompt",
                    "final_prompt": build_final_prompt(prompt_item["writing_prompt"], modifier),
                }
            )

        for variant in variants:
            group_index += 1
            prompt_text = build_prompt_text(tokenizer, args.thinking_mode, variant["final_prompt"])
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
            start = time.time()
            response = sampler.sample(
                prompt=types.ModelInput.from_ints(prompt_tokens),
                num_samples=args.num_samples,
                sampling_params=types.SamplingParams(
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                ),
            ).result()
            latency = time.time() - start

            for sample_index, seq in enumerate(response.sequences):
                response_text = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
                record = {
                    "record_id": str(uuid.uuid4()),
                    "prompt_id": prompt_item["prompt_id"],
                    "writing_prompt": prompt_item["writing_prompt"],
                    "seed_modifier_index": variant["seed_modifier_index"],
                    "seed_modifier": variant["seed_modifier"],
                    "variant_type": variant["variant_type"],
                    "final_prompt": variant["final_prompt"],
                    "sample_index": sample_index,
                    "response_text": response_text,
                    "error": None,
                    "created_at_utc": utc_now(),
                    "latency_seconds": latency,
                    "stop_reason": getattr(seq, "stop_reason", None),
                }
                doc["records"].append(record)
                print(
                    f"[tinker-generate] prompt={prompt_item['prompt_id']} "
                    f"mod={variant['seed_modifier_index']} sample={sample_index + 1}/{args.num_samples} "
                    f"status=ok latency_sec={latency:.3f}"
                )

            save_json(out_path, doc)
            print(
                f"[tinker-generate] group={group_index}/{total_groups} "
                f"prompt={prompt_item['prompt_id']} mod={variant['seed_modifier_index']} "
                f"saved={out_path}"
            )

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
