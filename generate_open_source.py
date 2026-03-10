#!/usr/bin/env python3
"""Generate Creative Writing Bench outputs with local/open-source HF models.

Writes the same generation JSON schema used by pipeline.py so downstream
embedding and accuracy scripts can reuse the outputs unchanged.
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import (
    DATASET_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    NUM_SAMPLES_PER_PROMPT,
    PROMPT_IDS_REQUESTED,
    build_final_prompt,
    load_prompt_items,
    slugify,
)


OUTPUT_DIR = Path("outputs/generations")
PRESET_MODEL_SPECS = {
    "tinker-baselines": [
        {
            "display_name": "qwen3-8b-non-reasoning",
            "provider": "huggingface",
            "api_model": "Qwen/Qwen3-8B",
            "thinking": "non_reasoning",
        },
        {
            "display_name": "qwen3-8b-reasoning",
            "provider": "huggingface",
            "api_model": "Qwen/Qwen3-8B",
            "thinking": "reasoning",
        },
        {
            "display_name": "llama-3-1-8b-instruct",
            "provider": "huggingface",
            "api_model": "meta-llama/Llama-3.1-8B-Instruct",
            "thinking": "non_reasoning",
        },
    ]
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_model_spec(spec: str) -> Dict[str, str]:
    parts = spec.split("|")
    if len(parts) != 3:
        raise ValueError(
            "Model spec must be alias|hf_model_id|thinking, "
            f"got: {spec}"
        )
    alias, model_id, thinking = [p.strip() for p in parts]
    if thinking not in {"reasoning", "non_reasoning"}:
        raise ValueError(f"thinking must be reasoning or non_reasoning, got: {thinking}")
    return {
        "display_name": alias,
        "provider": "huggingface",
        "api_model": model_id,
        "thinking": thinking,
    }


def build_prompt_text(tokenizer: Any, model_spec: Dict[str, str], prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Qwen3 supports explicit thinking toggling in its chat template.
        if "Qwen3" in model_spec["api_model"]:
            kwargs["enable_thinking"] = model_spec["thinking"] == "reasoning"
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Fallback for tokenizers that do not support enable_thinking.
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    return prompt


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def load_hf_model(model_id: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_one(
    tokenizer: Any,
    model: Any,
    model_spec: Dict[str, str],
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt_text = build_prompt_text(tokenizer, model_spec, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate outputs with local/open-source HF models")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help=(
            "Model spec in alias|hf_model_id|thinking format. "
            "Example: qwen2.5-7b|Qwen/Qwen2.5-7B-Instruct|non_reasoning"
        ),
    )
    parser.add_argument(
        "--preset",
        action="append",
        default=[],
        choices=sorted(PRESET_MODEL_SPECS.keys()),
        help="Predefined model set to generate",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max-modifiers", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES_PER_PROMPT)
    args = parser.parse_args()

    load_dotenv()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_items, missing = load_prompt_items(DATASET_PATH, PROMPT_IDS_REQUESTED, args.max_modifiers)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_specs = [parse_model_spec(spec) for spec in args.model_spec]
    for preset in args.preset:
        model_specs.extend(PRESET_MODEL_SPECS[preset])
    if not model_specs:
        raise ValueError("Provide at least one --model-spec or --preset")

    for model_spec in model_specs:
        display_name = model_spec["display_name"]
        model_id = model_spec["api_model"]
        out_path = OUTPUT_DIR / f"{slugify(display_name)}.json"

        print(f"[generate-open] loading model={display_name} hf_id={model_id}")
        tokenizer, model = load_hf_model(model_id)

        doc: Dict[str, Any] = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": model_spec,
            "dataset": {
                "path": str(DATASET_PATH),
                "prompt_ids_requested": PROMPT_IDS_REQUESTED,
                "missing_prompt_ids": missing,
            },
            "generation_config": {
                "num_samples_per_prompt": args.num_samples,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "records": [],
        }
        save_json(out_path, doc)

        for prompt_item in prompt_items:
            mods = prompt_item["seed_modifiers"]
            if not mods:
                continue

            for modifier_idx, modifier in enumerate(mods):
                final_prompt = build_final_prompt(prompt_item["writing_prompt"], modifier)
                for sample_idx in range(args.num_samples):
                    started = time.time()
                    record = {
                        "record_id": str(uuid.uuid4()),
                        "created_at_utc": utc_now(),
                        "prompt_id": prompt_item["prompt_id"],
                        "prompt_title": prompt_item["title"],
                        "prompt_category": prompt_item["category"],
                        "seed_modifier_index": modifier_idx,
                        "seed_modifier": modifier,
                        "sample_index": sample_idx,
                        "base_prompt": prompt_item["writing_prompt"],
                        "final_prompt": final_prompt,
                        "response_text": None,
                        "latency_sec": None,
                        "raw_api_response": None,
                        "error": None,
                    }
                    try:
                        record["response_text"] = generate_one(
                            tokenizer=tokenizer,
                            model=model,
                            model_spec=model_spec,
                            prompt=final_prompt,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_tokens,
                        )
                    except Exception as e:
                        record["error"] = str(e)

                    record["latency_sec"] = round(time.time() - started, 3)
                    doc["records"].append(record)
                    save_json(out_path, doc)
                    print(
                        f"[generate-open] model={display_name} prompt={prompt_item['prompt_id']} "
                        f"mod={modifier_idx} sample={sample_idx + 1}/{args.num_samples} "
                        f"status={'ok' if record['error'] is None else 'error'}"
                    )

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
