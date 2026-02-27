#!/usr/bin/env python3
"""Compute centroid-deviation metrics from cached embeddings.

Uses existing files:
- outputs/generations/*.json
- outputs/embeddings/*_embeddings.json

Writes per-model files:
- outputs/analysis/*_deviation_metrics.json

By default, cosine deviation is the primary metric for text embeddings.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

GEN_DIR = Path("outputs/generations")
EMB_DIR = Path("outputs/embeddings")
ANALYSIS_DIR = Path("outputs/analysis")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def group_key(rec: Dict[str, Any]) -> Tuple[int, int]:
    return int(rec["prompt_id"]), int(rec["seed_modifier_index"])


def compute_group_metrics(vectors: List[np.ndarray]) -> Dict[str, float]:
    centroid = np.mean(np.stack(vectors, axis=0), axis=0)
    cos_dists = [cosine_distance(v, centroid) for v in vectors]
    euc_dists = [euclidean_distance(v, centroid) for v in vectors]
    return {
        "avg_cosine_deviation": float(np.mean(cos_dists)),
        "max_cosine_deviation": float(np.max(cos_dists)),
        "avg_euclidean_deviation": float(np.mean(euc_dists)),
        "max_euclidean_deviation": float(np.max(euc_dists)),
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found in outputs/generations")

    index_doc: Dict[str, Any] = {
        "note": "For text embeddings, cosine deviation is usually the primary metric.",
        "files": [],
    }

    for gen_path in model_files:
        stem = gen_path.stem
        emb_path = EMB_DIR / f"{stem}_embeddings.json"
        out_path = ANALYSIS_DIR / f"{stem}_deviation_metrics.json"

        if not emb_path.exists():
            print(f"[skip] missing embedding cache: {emb_path}")
            continue

        with gen_path.open("r", encoding="utf-8") as f:
            gen_doc = json.load(f)
        with emb_path.open("r", encoding="utf-8") as f:
            emb_doc = json.load(f)

        model_name = gen_doc.get("model", {}).get("display_name", stem)
        emb_entries = emb_doc.get("entries", {})

        grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for rec in gen_doc.get("records", []):
            if rec.get("error") is not None:
                continue
            if not rec.get("response_text"):
                continue
            rid = rec.get("record_id")
            if rid not in emb_entries:
                continue
            grouped.setdefault(group_key(rec), []).append(rec)

        result: Dict[str, Any] = {
            "model": model_name,
            "source_generation_file": str(gen_path),
            "source_embedding_file": str(emb_path),
            "distance_recommendation": "cosine",
            "groups": [],
            "summary": {},
        }

        avg_cos_values: List[float] = []
        max_cos_values: List[float] = []
        avg_euc_values: List[float] = []
        max_euc_values: List[float] = []

        for (prompt_id, modifier_idx), records in sorted(grouped.items()):
            if len(records) < 2:
                continue
            records = sorted(records, key=lambda r: int(r["sample_index"]))
            vectors = [
                np.array(emb_entries[r["record_id"]]["embedding"], dtype=np.float64)
                for r in records
            ]

            metrics = compute_group_metrics(vectors)
            avg_cos_values.append(metrics["avg_cosine_deviation"])
            max_cos_values.append(metrics["max_cosine_deviation"])
            avg_euc_values.append(metrics["avg_euclidean_deviation"])
            max_euc_values.append(metrics["max_euclidean_deviation"])

            result["groups"].append(
                {
                    "prompt_id": prompt_id,
                    "seed_modifier_index": modifier_idx,
                    "num_samples": len(records),
                    **metrics,
                    "sample_record_ids": [r["record_id"] for r in records],
                }
            )

        result["summary"] = {
            "num_groups": len(result["groups"]),
            "mean_avg_cosine_deviation": float(np.mean(avg_cos_values)) if avg_cos_values else None,
            "mean_max_cosine_deviation": float(np.mean(max_cos_values)) if max_cos_values else None,
            "mean_avg_euclidean_deviation": float(np.mean(avg_euc_values)) if avg_euc_values else None,
            "mean_max_euclidean_deviation": float(np.mean(max_euc_values)) if max_euc_values else None,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")

        index_doc["files"].append(
            {
                "model": model_name,
                "metrics_file": str(out_path),
                "num_groups": result["summary"]["num_groups"],
            }
        )

    index_path = ANALYSIS_DIR / "deviation_metrics_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(index_doc, f, ensure_ascii=False, indent=2)
    print(f"[saved] {index_path}")


if __name__ == "__main__":
    main()
