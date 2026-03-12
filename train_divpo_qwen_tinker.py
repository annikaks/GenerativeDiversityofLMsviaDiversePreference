#!/usr/bin/env python3
"""Train Qwen3-8B DivPO adapters on Tinker from prebuilt pairwise data.

Expected input:
- `preference_pairs.jsonl` files produced by `build_divpo_pairs.py`

This script keeps the pair-construction logic local/Modal and moves the actual
LoRA fine-tuning to Tinker. Training uses a custom DPO loss on top of Tinker's
LoRA training client.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

import tinker
from tinker import types


MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_OUTPUT_ROOT = Path("outputs/divpo")


def make_safe_weights_label(*parts: str) -> str:
    raw = "-".join(parts)
    safe = []
    for ch in raw:
        if ch.isalnum() or ch in "-_.":
            safe.append(ch)
        else:
            safe.append("-")
    return "".join(safe)


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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def build_sequence_tokens(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    tokenizer.truncation_side = "left"
    prompt_tokens = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_tokens = tokenizer.encode(
        prompt + completion,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    completion_mask = [0.0] * max(len(full_tokens) - 1, 0)
    completion_start = max(len(prompt_tokens) - 1, 0)
    for idx in range(completion_start, len(completion_mask)):
        completion_mask[idx] = 1.0
    return full_tokens, completion_mask


def compute_reference_logprob_sum(
    sampler: Any,
    tokens: List[int],
    prompt_token_count: int,
) -> float:
    logprobs = sampler.compute_logprobs(types.ModelInput.from_ints(tokens)).result()
    values = [lp for lp in logprobs[prompt_token_count:] if lp is not None]
    return float(sum(values))


def annotate_pairs_with_reference(
    rows: List[Dict[str, Any]],
    tokenizer: Any,
    sampler: Any,
    max_length: int,
    checkpoint_path: Path,
    checkpoint_every: int,
    source_pairs_path: Path,
    initial_rows: List[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = list(initial_rows or [])
    start_time = time.time()
    total = len(rows)
    start_index = len(annotated)
    if start_index > 0:
        print(
            f"[tinker-divpo] resuming annotated pairs from {checkpoint_path} "
            f"completed={start_index}/{total}"
        )
    for idx, row in enumerate(rows[start_index:], start=start_index + 1):
        prompt = row["prompt"]
        chosen = row["chosen"]
        rejected = row["rejected"]

        chosen_tokens, chosen_mask = build_sequence_tokens(tokenizer, prompt, chosen, max_length)
        rejected_tokens, rejected_mask = build_sequence_tokens(tokenizer, prompt, rejected, max_length)
        prompt_token_count = len(
            tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
        )

        chosen_ref = compute_reference_logprob_sum(sampler, chosen_tokens, prompt_token_count)
        rejected_ref = compute_reference_logprob_sum(sampler, rejected_tokens, prompt_token_count)

        annotated.append(
            {
                **row,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "chosen_mask": chosen_mask,
                "rejected_mask": rejected_mask,
                "chosen_ref_logprob_sum": chosen_ref,
                "rejected_ref_logprob_sum": rejected_ref,
            }
        )
        if idx % checkpoint_every == 0 or idx == total:
            save_json(
                checkpoint_path,
                {
                    "rows": annotated,
                    "source_pairs_path": str(source_pairs_path),
                    "total_rows": total,
                    "completed_rows": len(annotated),
                    "is_complete": len(annotated) == total,
                    "created_at_utc": "generated-by-train_divpo_qwen_tinker",
                },
            )
            print(
                f"[tinker-divpo] annotation_checkpoint completed={len(annotated)}/{total} "
                f"path={checkpoint_path}"
            )
        if idx == 1 or idx % 10 == 0 or idx == total:
            elapsed_min = (time.time() - start_time) / 60.0
            per_pair_sec = (time.time() - start_time) / max(idx - start_index, 1)
            remaining = total - idx
            eta_min = (remaining * per_pair_sec) / 60.0
            print(
                f"[tinker-divpo] annotated_pairs={idx}/{total} "
                f"elapsed_min={elapsed_min:.1f} eta_min~{eta_min:.1f}"
            )
    return annotated


def make_dpo_batch(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[tinker.Datum], List[Dict[str, Any]]]:
    batch: List[tinker.Datum] = []
    datum_meta: List[Dict[str, Any]] = []
    for pair_index, row in enumerate(rows):
        batch.append(
            tinker.Datum(
                model_input=types.ModelInput.from_ints(row["chosen_tokens"][:-1]),
                loss_fn_inputs={
                    "target_tokens": row["chosen_tokens"][1:],
                    "weights": [1.0] * (len(row["chosen_tokens"]) - 1),
                },
            )
        )
        datum_meta.append(
            {
                "completion_mask": row["chosen_mask"],
                "reference_logprob_sum": row["chosen_ref_logprob_sum"],
                "pair_index": pair_index,
                "is_chosen": 1,
            }
        )
        batch.append(
            tinker.Datum(
                model_input=types.ModelInput.from_ints(row["rejected_tokens"][:-1]),
                loss_fn_inputs={
                    "target_tokens": row["rejected_tokens"][1:],
                    "weights": [1.0] * (len(row["rejected_tokens"]) - 1),
                },
            )
        )
        datum_meta.append(
            {
                "completion_mask": row["rejected_mask"],
                "reference_logprob_sum": row["rejected_ref_logprob_sum"],
                "pair_index": pair_index,
                "is_chosen": 0,
            }
        )
    return batch, datum_meta


def dpo_loss_factory(beta: float, datum_meta: Sequence[Dict[str, Any]]):
    def loss_fn(data: List[tinker.Datum], logprobs: List[torch.Tensor]):
        pair_state: Dict[int, Dict[str, torch.Tensor]] = {}
        device = logprobs[0].device

        for meta, seq_logprobs in zip(datum_meta, logprobs):
            mask = torch.as_tensor(meta["completion_mask"], device=device)
            ref = torch.as_tensor(meta["reference_logprob_sum"], device=device)
            pair_index = int(meta["pair_index"])
            is_chosen = int(meta["is_chosen"])

            current_sum = (seq_logprobs * mask).sum()
            state = pair_state.setdefault(pair_index, {})
            if is_chosen == 1:
                state["chosen_sum"] = current_sum
                state["chosen_ref"] = ref
            else:
                state["rejected_sum"] = current_sum
                state["rejected_ref"] = ref

        losses: List[torch.Tensor] = []
        reward_margins: List[torch.Tensor] = []
        for state in pair_state.values():
            logit = beta * (
                (state["chosen_sum"] - state["chosen_ref"])
                - (state["rejected_sum"] - state["rejected_ref"])
            )
            losses.append(-torch.nn.functional.logsigmoid(logit))
            reward_margins.append(logit)

        loss = torch.stack(losses).mean()
        metrics = {
            "pair_loss": float(loss.detach().cpu().item()),
            "reward_margin": float(torch.stack(reward_margins).mean().detach().cpu().item()),
            "pairs_in_batch": float(len(pair_state)),
        }
        return loss, metrics

    return loss_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-8B DivPO adapters on Tinker")
    parser.add_argument(
        "--thinking-mode",
        choices=["non_reasoning", "reasoning"],
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pairs-path", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size-pairs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--save-every-steps", type=int, default=5)
    parser.add_argument("--annotation-checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    default_mode_dir = args.output_root / f"qwen3-8b-{args.thinking_mode}-divpo"
    pairs_path = args.pairs_path or (default_mode_dir / "preference_pairs.jsonl")
    source_mode_dir = pairs_path.parent if args.pairs_path is not None else default_mode_dir
    run_dir = source_mode_dir / args.run_name if args.run_name else source_mode_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    annotated_pairs_path = source_mode_dir / "preference_pairs_tinker_ready.json"
    metrics_path = run_dir / "tinker_training_metrics.json"
    checkpoints_path = run_dir / "tinker_checkpoints.json"

    pair_rows = load_jsonl(pairs_path)
    if not pair_rows:
        raise ValueError(f"No pair rows found in {pairs_path}")
    print(f"[tinker-divpo] loaded_pairs={len(pair_rows)} from {pairs_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    service_client = tinker.ServiceClient()
    reference_sampler = service_client.create_sampling_client(base_model=MODEL_ID)
    print(f"[tinker-divpo] connected base_model={MODEL_ID} mode={args.thinking_mode}")

    existing_annotated_rows: List[Dict[str, Any]] = []
    annotated_doc: Dict[str, Any] | None = None
    if annotated_pairs_path.exists():
        with annotated_pairs_path.open("r", encoding="utf-8") as f:
            annotated_doc = json.load(f)
        existing_annotated_rows = annotated_doc.get("rows", [])
    if annotated_doc and annotated_doc.get("is_complete") and len(existing_annotated_rows) == len(pair_rows):
        annotated_rows = existing_annotated_rows
        print(
            f"[tinker-divpo] reusing annotated pairs from {annotated_pairs_path} "
            f"rows={len(annotated_rows)}"
        )
    else:
        print(f"[tinker-divpo] annotating {len(pair_rows)} pairs with reference logprobs")
        annotated_rows = annotate_pairs_with_reference(
            rows=pair_rows,
            tokenizer=tokenizer,
            sampler=reference_sampler,
            max_length=args.max_length,
            checkpoint_path=annotated_pairs_path,
            checkpoint_every=args.annotation_checkpoint_every,
            source_pairs_path=pairs_path,
            initial_rows=existing_annotated_rows,
        )
        print(f"[tinker-divpo] saved annotated pairs to {annotated_pairs_path}")

    training_client = service_client.create_lora_training_client(
        base_model=MODEL_ID,
        rank=args.rank,
    )
    print(
        f"[tinker-divpo] training_start num_steps={args.num_steps} "
        f"batch_size_pairs={args.batch_size_pairs} save_every_steps={args.save_every_steps}"
    )

    metrics: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    random.shuffle(annotated_rows)
    row_batches = list(batched(annotated_rows, args.batch_size_pairs))
    started = time.time()
    for step in range(args.num_steps):
        rows = row_batches[step % len(row_batches)]
        data, datum_meta = make_dpo_batch(rows)
        loss_fn = dpo_loss_factory(args.beta, datum_meta)
        loss_result = training_client.forward_backward_custom(data, loss_fn).result()
        training_client.optim_step(
            types.AdamParams(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        ).result()

        step_metrics = {
            "step": step + 1,
            "loss": float(loss_result.metrics.get("pair_loss", 0.0)),
            "reward_margin": float(loss_result.metrics.get("reward_margin", 0.0)),
            "pairs_in_batch": int(loss_result.metrics.get("pairs_in_batch", 0)),
        }
        metrics.append(step_metrics)
        elapsed_min = (time.time() - started) / 60.0
        steps_done = step + 1
        remaining_steps = args.num_steps - steps_done
        eta_min = (elapsed_min / steps_done) * remaining_steps if steps_done > 0 else 0.0
        print(
            f"[tinker-divpo] step={step + 1}/{args.num_steps} "
            f"loss={step_metrics['loss']:.4f} reward_margin={step_metrics['reward_margin']:.4f} "
            f"elapsed_min={elapsed_min:.1f} eta_min~{eta_min:.1f}"
        )

        if (step + 1) % args.save_every_steps == 0:
            checkpoint_name = make_safe_weights_label(
                source_mode_dir.name,
                run_dir.name,
                f"checkpoint-step-{step + 1:04d}",
            )
            checkpoint_path = training_client.save_state(checkpoint_name).result().path
            checkpoint_obj = {
                "step": step + 1,
                "checkpoint_name": checkpoint_name,
                "checkpoint_path": checkpoint_path,
            }
            checkpoints.append(checkpoint_obj)
            print(
                f"[tinker-divpo] checkpoint_saved step={step + 1}/{args.num_steps} "
                f"path={checkpoint_path}"
            )
            save_json(
                checkpoints_path,
                {
                    "created_at_utc": "generated-by-train_divpo_qwen_tinker",
                    "base_model": MODEL_ID,
                    "thinking_mode": args.thinking_mode,
                    "checkpoints": checkpoints,
                },
            )
            save_json(
                metrics_path,
                {
                    "created_at_utc": "generated-by-train_divpo_qwen_tinker",
                    "last_checkpoint_path": checkpoint_path,
                    "last_checkpoint_name": checkpoint_name,
                    "num_checkpoints_saved": len(checkpoints),
                    "metrics": metrics,
                },
            )

    final_checkpoint_name = make_safe_weights_label(source_mode_dir.name, run_dir.name, "final-state")
    final_state = training_client.save_state(final_checkpoint_name).result().path
    final_sampler_checkpoint_name = make_safe_weights_label(
        source_mode_dir.name, run_dir.name, "final-sampler"
    )
    final_sampler_checkpoint_path = training_client.save_weights_for_sampler(
        final_sampler_checkpoint_name
    ).result().path
    try:
        final_sampler = training_client.save_weights_and_get_sampling_client()
        final_sampler_repr = str(final_sampler)
    except Exception as exc:
        final_sampler_repr = f"ERROR: {exc}"
    checkpoints.append(
        {
            "step": args.num_steps,
            "checkpoint_name": final_checkpoint_name,
            "checkpoint_path": final_state,
            "kind": "final_state",
        }
    )
    checkpoints.append(
        {
            "step": args.num_steps,
            "checkpoint_name": final_sampler_checkpoint_name,
            "checkpoint_path": final_sampler_checkpoint_path,
            "kind": "final_sampler_state",
        }
    )
    save_json(
        checkpoints_path,
        {
            "created_at_utc": "generated-by-train_divpo_qwen_tinker",
            "base_model": MODEL_ID,
            "thinking_mode": args.thinking_mode,
            "checkpoints": checkpoints,
            "final_checkpoint_name": final_checkpoint_name,
            "final_state_path": final_state,
            "final_sampler_checkpoint_name": final_sampler_checkpoint_name,
            "final_sampler_checkpoint_path": final_sampler_checkpoint_path,
        },
    )
    save_json(
        metrics_path,
        {
            "created_at_utc": "generated-by-train_divpo_qwen_tinker",
            "final_checkpoint_name": final_checkpoint_name,
            "final_state_path": final_state,
            "final_sampler_checkpoint_name": final_sampler_checkpoint_name,
            "final_sampler_checkpoint_path": final_sampler_checkpoint_path,
            "sampling_client": final_sampler_repr,
            "num_checkpoints_saved": len(checkpoints),
            "metrics": metrics,
        },
    )
    print(f"[tinker-divpo] final_state={final_state}")
    print(f"[tinker-divpo] final_sampler_checkpoint={final_sampler_checkpoint_path}")


if __name__ == "__main__":
    main()
