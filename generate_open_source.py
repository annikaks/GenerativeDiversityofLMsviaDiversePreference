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
DEFAULT_NUM_SAMPLES = 4
DEFAULT_NUM_SEED_MODIFIERS = 2


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


def build_prompt_variants(prompt_item: Dict[str, Any], max_seed_modifiers: int) -> List[Dict[str, Any]]:
    """Return prompt variants: base prompt + first N seed-modified prompts."""
    variants: List[Dict[str, Any]] = [
        {
            "seed_modifier_index": -1,
            "seed_modifier": "",
            "variant_type": "base_prompt",
            "final_prompt": prompt_item["writing_prompt"],
        }
    ]
    for modifier_idx, modifier in enumerate(prompt_item["seed_modifiers"][:max_seed_modifiers]):
        variants.append(
            {
                "seed_modifier_index": modifier_idx,
                "seed_modifier": modifier,
                "variant_type": "seed_modified_prompt",
                "final_prompt": build_final_prompt(prompt_item["writing_prompt"], modifier),
            }
        )
    return variants


def generate_one(
    tokenizer: Any,
    model: Any,
    model_spec: Dict[str, str],
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    return generate_batch(
        tokenizer=tokenizer,
        model=model,
        model_spec=model_spec,
        prompts=[prompt],
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )[0]


def generate_batch(
    tokenizer: Any,
    model: Any,
    model_spec: Dict[str, str],
    prompts: List[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[str]:
    prompt_texts = [build_prompt_text(tokenizer, model_spec, prompt) for prompt in prompts]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    results: List[str] = []
    for idx, input_len in enumerate(input_lengths):
        new_tokens = outputs[idx][int(input_len) :]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


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
    parser.add_argument(
        "--max-modifiers",
        type=int,
        default=DEFAULT_NUM_SEED_MODIFIERS,
        help="Number of seed-modified variants to use in addition to the unmodified base prompt",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=4)
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

        pending: List[Dict[str, Any]] = []
        for prompt_item in prompt_items:
            variants = build_prompt_variants(prompt_item, args.max_modifiers)
            for variant in variants:
                for sample_idx in range(args.num_samples):
                    pending.append(
                        {
                            "record_id": str(uuid.uuid4()),
                            "created_at_utc": utc_now(),
                            "prompt_id": prompt_item["prompt_id"],
                            "prompt_title": prompt_item["title"],
                            "prompt_category": prompt_item["category"],
                            "seed_modifier_index": variant["seed_modifier_index"],
                            "seed_modifier": variant["seed_modifier"],
                            "variant_type": variant["variant_type"],
                            "sample_index": sample_idx,
                            "base_prompt": prompt_item["writing_prompt"],
                            "final_prompt": variant["final_prompt"],
                            "response_text": None,
                            "latency_sec": None,
                            "raw_api_response": None,
                            "error": None,
                        }
                    )

        total = len(pending)
        for start_idx in range(0, total, args.batch_size):
            batch = pending[start_idx : start_idx + args.batch_size]
            started = time.time()
            prompts = [item["final_prompt"] for item in batch]
            try:
                responses = generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    model_spec=model_spec,
                    prompts=prompts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_tokens,
                )
                for item, response_text in zip(batch, responses):
                    item["response_text"] = response_text
            except Exception as e:
                for item in batch:
                    item["error"] = str(e)

            batch_latency = round(time.time() - started, 3)
            for item in batch:
                item["latency_sec"] = batch_latency
                doc["records"].append(item)
                print(
                    f"[generate-open] model={display_name} prompt={item['prompt_id']} "
                    f"mod={item['seed_modifier_index']} sample={item['sample_index'] + 1}/{args.num_samples} "
                    f"status={'ok' if item['error'] is None else 'error'} "
                    f"latency_sec={item['latency_sec']}"
                )
            save_json(out_path, doc)
            print(
                f"[generate-open] model={display_name} batch={min(start_idx + len(batch), total)}/{total} "
                f"batch_latency_sec={batch_latency} saved={out_path}"
            )

        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
