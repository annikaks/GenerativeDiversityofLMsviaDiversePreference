#!/usr/bin/env python3
"""Build DivPO preference data and train Qwen3-8B LoRA adapters.

This script implements a practical DivPO-style workflow for the Creative
Writing Bench training split:

1. Use prompts 0..26 and all 10 seed modifiers.
2. Generate multiple samples per prompt-condition in batches.
3. Embed prompts/responses and compute:
   - a prompt-adherence proxy from prompt/response cosine similarity
   - a marginal diversity contribution relative to a selected set
4. Convert those set-aware rewards into pairwise preferences.
5. Train a LoRA adapter with TRL's DPOTrainer, which handles reference
   log-prob precomputation internally.

The resulting adapters are intended as the first concrete post-training step
for the Qwen3 reasoning vs non-reasoning comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from generate_open_source import build_prompt_text
from pipeline import (
    DATASET_PATH,
    build_final_prompt,
    fetch_openai_embeddings,
    load_prompt_items,
    slugify,
)


TRAIN_PROMPT_IDS = list(range(0, 27))
MODEL_ID = "Qwen/Qwen3-8B"
THINKING_MODES = ("non_reasoning", "reasoning")
DEFAULT_OUTPUT_ROOT = Path("outputs/divpo")


@dataclass
class CandidateRecord:
    prompt_id: int
    seed_modifier_index: int
    sample_index: int
    prompt_text: str
    formatted_prompt: str
    response_text: str
    quality_score: float
    prompt_response_cosine: float
    distinct2: float
    embedding: List[float]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(1.0 - np.dot(vec_a, vec_b))


def distinct_2(text: str) -> float:
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return len(set(bigrams)) / float(len(bigrams))


def load_train_prompt_items() -> Tuple[List[Dict[str, Any]], List[int]]:
    return load_prompt_items(DATASET_PATH, TRAIN_PROMPT_IDS, max_modifiers=10)


def build_training_prompt_variants(prompt_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = [
        {
            "seed_modifier_index": -1,
            "seed_modifier": "",
            "variant_type": "base_prompt",
            "final_prompt": prompt_item["writing_prompt"],
        }
    ]
    for modifier_idx, modifier in enumerate(prompt_item["seed_modifiers"]):
        variants.append(
            {
                "seed_modifier_index": modifier_idx,
                "seed_modifier": modifier,
                "variant_type": "seed_modified_prompt",
                "final_prompt": build_final_prompt(prompt_item["writing_prompt"], modifier),
            }
        )
    return variants


def load_qwen_model(model_id: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")
    return tokenizer, model


def generate_batch(
    tokenizer: Any,
    model: Any,
    thinking_mode: str,
    prompts: Sequence[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> List[str]:
    model_spec = {"api_model": MODEL_ID, "thinking": thinking_mode}
    prompt_texts = [build_prompt_text(tokenizer, model_spec, prompt) for prompt in prompts]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
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


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def score_candidate_group(
    prompt_text: str,
    formatted_prompt: str,
    responses: Sequence[str],
    embedding_batch_size: int,
) -> List[CandidateRecord]:
    texts = [prompt_text] + list(responses)
    vectors: List[List[float]] = []
    for chunk in batched(texts, embedding_batch_size):
        vectors.extend(fetch_openai_embeddings(list(chunk), "text-embedding-3-large"))

    matrix = normalize_rows(np.asarray(vectors, dtype=np.float32))
    prompt_vec = matrix[0]
    response_vecs = matrix[1:]

    prompt_sims = response_vecs @ prompt_vec
    distinct_scores = np.asarray([distinct_2(resp) for resp in responses], dtype=np.float32)

    if distinct_scores.max() - distinct_scores.min() < 1e-8:
        distinct_scaled = np.zeros_like(distinct_scores)
    else:
        distinct_scaled = (distinct_scores - distinct_scores.min()) / (
            distinct_scores.max() - distinct_scores.min()
        )

    # Quality is primarily prompt adherence, with a small lexical diversity term
    # to avoid over-rewarding repetitive but on-topic responses.
    quality = 0.85 * prompt_sims + 0.15 * distinct_scaled

    rows: List[CandidateRecord] = []
    for sample_index, response in enumerate(responses):
        rows.append(
            CandidateRecord(
                prompt_id=-1,
                seed_modifier_index=-1,
                sample_index=sample_index,
                prompt_text=prompt_text,
                formatted_prompt=formatted_prompt,
                response_text=response,
                quality_score=float(quality[sample_index]),
                prompt_response_cosine=float(prompt_sims[sample_index]),
                distinct2=float(distinct_scores[sample_index]),
                embedding=response_vecs[sample_index].tolist(),
            )
        )
    return rows


def build_preference_pairs_for_group(
    group_rows: List[CandidateRecord],
    prompt_id: int,
    seed_modifier_index: int,
    quality_floor_quantile: float,
    lambda_quality: float,
    lambda_diversity: float,
    subset_size: int,
    negatives_per_step: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not group_rows:
        return [], {
            "prompt_id": prompt_id,
            "seed_modifier_index": seed_modifier_index,
            "num_candidates": 0,
            "num_valid_candidates": 0,
            "num_pairs": 0,
        }

    qualities = np.asarray([row.quality_score for row in group_rows], dtype=np.float32)
    quality_floor = float(np.quantile(qualities, quality_floor_quantile))
    valid_indices = [idx for idx, row in enumerate(group_rows) if row.quality_score >= quality_floor]
    if len(valid_indices) < 2:
        return [], {
            "prompt_id": prompt_id,
            "seed_modifier_index": seed_modifier_index,
            "num_candidates": len(group_rows),
            "num_valid_candidates": len(valid_indices),
            "num_pairs": 0,
        }

    selected: List[int] = []
    unselected = set(valid_indices)
    pairs: List[Dict[str, Any]] = []

    while unselected and len(selected) < min(subset_size, len(valid_indices)):
        rewards: Dict[int, float] = {}
        diversity_scores: Dict[int, float] = {}
        for idx in unselected:
            row = group_rows[idx]
            if not selected:
                marginal_diversity = 0.0
            else:
                distances = [
                    cosine_distance(
                        np.asarray(row.embedding, dtype=np.float32),
                        np.asarray(group_rows[selected_idx].embedding, dtype=np.float32),
                    )
                    for selected_idx in selected
                ]
                marginal_diversity = float(np.mean(distances))
            diversity_scores[idx] = marginal_diversity
            rewards[idx] = (
                lambda_quality * row.quality_score
                + lambda_diversity * marginal_diversity
            )

        chosen_idx = max(rewards, key=rewards.get)
        chosen_row = group_rows[chosen_idx]
        selected.append(chosen_idx)
        unselected.remove(chosen_idx)

        negatives = sorted(unselected, key=lambda idx: rewards[idx])[:negatives_per_step]
        for rejected_idx in negatives:
            rejected_row = group_rows[rejected_idx]
            pairs.append(
                {
                    "prompt_id": prompt_id,
                    "seed_modifier_index": seed_modifier_index,
                    "prompt": chosen_row.formatted_prompt,
                    "chosen": chosen_row.response_text,
                    "rejected": rejected_row.response_text,
                    "chosen_quality": chosen_row.quality_score,
                    "rejected_quality": rejected_row.quality_score,
                    "chosen_marginal_diversity": diversity_scores[chosen_idx],
                    "rejected_marginal_diversity": diversity_scores[rejected_idx],
                    "reward_gap": rewards[chosen_idx] - rewards[rejected_idx],
                    "pair_source": "greedy_marginal_diversity",
                }
            )

    summary = {
        "prompt_id": prompt_id,
        "seed_modifier_index": seed_modifier_index,
        "num_candidates": len(group_rows),
        "num_valid_candidates": len(valid_indices),
        "num_pairs": len(pairs),
        "quality_floor": quality_floor,
    }
    return pairs, summary


def build_preference_dataset(
    thinking_mode: str,
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    samples_per_prompt: int,
    generation_batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    embedding_batch_size: int,
    quality_floor_quantile: float,
    lambda_quality: float,
    lambda_diversity: float,
    subset_size: int,
    negatives_per_step: int,
) -> Path:
    prompt_items, missing = load_train_prompt_items()
    candidates_path = output_dir / "candidates.jsonl"
    pairs_path = output_dir / "preference_pairs.jsonl"
    summary_path = output_dir / "pair_build_summary.json"

    existing_candidates = load_jsonl(candidates_path)
    group_counts: Dict[Tuple[int, int], int] = {}
    for row in existing_candidates:
        key = (row["prompt_id"], row["seed_modifier_index"])
        group_counts[key] = group_counts.get(key, 0) + 1
    completed_groups = {
        key for key, count in group_counts.items() if count >= samples_per_prompt
    }
    print(
        f"[divpo] mode={thinking_mode} prompts={len(prompt_items)} missing={missing} "
        f"completed_groups={len(completed_groups)}"
    )

    group_summaries: List[Dict[str, Any]] = []
    total_pairs = 0
    total_expected_groups = sum(len(build_training_prompt_variants(prompt_item)) for prompt_item in prompt_items)
    completed_groups_before = len(completed_groups)
    completed_groups_now = completed_groups_before
    run_started = time.time()
    for prompt_item in prompt_items:
        variants = build_training_prompt_variants(prompt_item)
        for variant in variants:
            group_key = (prompt_item["prompt_id"], variant["seed_modifier_index"])
            if group_key in completed_groups:
                print(
                    f"[divpo] mode={thinking_mode} prompt={prompt_item['prompt_id']} "
                    f"mod={variant['seed_modifier_index']} status=skip existing_pairs"
                )
                continue

            prompts = [variant["final_prompt"]] * samples_per_prompt
            responses: List[str] = []
            started = time.time()
            for chunk in batched(prompts, generation_batch_size):
                responses.extend(
                    generate_batch(
                        tokenizer=tokenizer,
                        model=model,
                        thinking_mode=thinking_mode,
                        prompts=chunk,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                    )
                )

            scored_rows = score_candidate_group(
                prompt_text=variant["final_prompt"],
                formatted_prompt=build_prompt_text(
                    tokenizer,
                    {"api_model": MODEL_ID, "thinking": thinking_mode},
                    variant["final_prompt"],
                ),
                responses=responses,
                embedding_batch_size=embedding_batch_size,
            )
            for row in scored_rows:
                row.prompt_id = prompt_item["prompt_id"]
                row.seed_modifier_index = variant["seed_modifier_index"]

            append_jsonl(
                candidates_path,
                [
                    {
                        "prompt_id": row.prompt_id,
                        "seed_modifier_index": row.seed_modifier_index,
                        "sample_index": row.sample_index,
                        "prompt_text": row.prompt_text,
                        "formatted_prompt": row.formatted_prompt,
                        "response_text": row.response_text,
                        "quality_score": row.quality_score,
                        "prompt_response_cosine": row.prompt_response_cosine,
                        "distinct2": row.distinct2,
                        "embedding": row.embedding,
                    }
                    for row in scored_rows
                ],
            )

            pairs, group_summary = build_preference_pairs_for_group(
                group_rows=scored_rows,
                prompt_id=prompt_item["prompt_id"],
                seed_modifier_index=variant["seed_modifier_index"],
                quality_floor_quantile=quality_floor_quantile,
                lambda_quality=lambda_quality,
                lambda_diversity=lambda_diversity,
                subset_size=subset_size,
                negatives_per_step=negatives_per_step,
            )
            if pairs:
                append_jsonl(pairs_path, pairs)
            total_pairs += len(pairs)
            completed_groups_now += 1
            elapsed_sec = max(time.time() - run_started, 1e-6)
            groups_done_this_run = completed_groups_now - completed_groups_before
            groups_remaining = total_expected_groups - completed_groups_now
            avg_sec_per_group = elapsed_sec / max(groups_done_this_run, 1)
            eta_sec = avg_sec_per_group * max(groups_remaining, 0)
            group_summary["generation_latency_sec"] = round(time.time() - started, 3)
            group_summary["completed_groups_global"] = completed_groups_now
            group_summary["total_expected_groups"] = total_expected_groups
            group_summaries.append(group_summary)
            save_json(
                summary_path,
                {
                    "created_at_utc": utc_now(),
                    "thinking_mode": thinking_mode,
                    "model_id": MODEL_ID,
                    "train_prompt_ids_requested": TRAIN_PROMPT_IDS,
                    "missing_prompt_ids": missing,
                    "samples_per_prompt": samples_per_prompt,
                    "total_pairs_so_far": total_pairs,
                    "groups_completed_this_run": len(group_summaries),
                    "groups_completed_global": completed_groups_now,
                    "total_expected_groups": total_expected_groups,
                    "group_summaries": group_summaries,
                },
            )
            print(
                f"[divpo] mode={thinking_mode} prompt={prompt_item['prompt_id']} "
                f"mod={variant['seed_modifier_index']} pairs={len(pairs)} "
                f"group={completed_groups_now}/{total_expected_groups} "
                f"latency_sec={group_summary['generation_latency_sec']} "
                f"elapsed_min={elapsed_sec / 60:.1f} eta_min~{eta_sec / 60:.1f}"
            )

    return pairs_path


def load_training_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)["rows"]
    return load_jsonl(path)


def apply_lora(model: Any) -> Any:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_with_trl_dpo(
    pair_rows: List[Dict[str, Any]],
    output_dir: Path,
    tokenizer: Any,
    model: Any,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    beta: float,
    num_epochs: int,
    max_length: int,
    save_every_steps: int,
) -> None:
    if not pair_rows:
        raise ValueError("No preference pairs available for training")
    dataset_rows = [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in pair_rows
    ]
    train_dataset = Dataset.from_list(dataset_rows)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    training_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
        logging_steps=5,
        save_steps=save_every_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        precompute_ref_log_probs=True,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    save_json(
        output_dir / "training_metrics.json",
        {
            "created_at_utc": utc_now(),
            "train_runtime_sec": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
            "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
            "num_pairs": len(pair_rows),
        },
    )
    print(f"[divpo-train] saved adapter to {output_dir / 'final_adapter'}")


def run_for_mode(args: argparse.Namespace, thinking_mode: str) -> None:
    mode_slug = slugify(f"qwen3-8b-{thinking_mode}-divpo")
    output_dir = args.output_root / mode_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[divpo] loading model={MODEL_ID} mode={thinking_mode}")
    tokenizer, model = load_qwen_model(MODEL_ID)

    if args.stage in {"all", "build-data"}:
        pairs_path = build_preference_dataset(
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
    else:
        pairs_path = output_dir / "preference_pairs.jsonl"

    if args.stage in {"all", "precompute-ref"}:
        print(
            "[divpo] explicit precompute-ref stage is no longer separate; "
            "TRL DPOTrainer will precompute reference log-probs during training."
        )

    if args.stage in {"all", "train", "precompute-ref"}:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer, model = load_qwen_model(MODEL_ID)
        model = apply_lora(model)
        train_rows = load_training_rows(pairs_path)
        train_with_trl_dpo(
            pair_rows=train_rows,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            beta=args.beta,
            num_epochs=args.num_epochs,
            max_length=args.max_length,
            save_every_steps=args.save_every_steps,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DivPO pairs and train Qwen3-8B adapters")
    parser.add_argument(
        "--thinking-mode",
        choices=["non_reasoning", "reasoning", "both"],
        default="both",
        help="Which Qwen3 mode to process",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "build-data", "precompute-ref", "train"],
        default="all",
        help="Pipeline stage to run",
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
    parser.add_argument("--reference-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--save-every-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    args.output_root.mkdir(parents=True, exist_ok=True)

    modes = THINKING_MODES if args.thinking_mode == "both" else (args.thinking_mode,)
    for thinking_mode in modes:
        run_for_mode(args, thinking_mode)


if __name__ == "__main__":
    main()
