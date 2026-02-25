#!/usr/bin/env python3
"""Diversity pipeline for Creative Writing Bench prompts.

This script supports:
1) Generation across multiple providers/models.
2) Embedding-space diversity analysis.
3) LLM-as-judge placeholder artifact generation.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

import numpy as np
from dotenv import load_dotenv


DATASET_PATH = Path("creative_writing_prompts_v3.json")
OUTPUT_DIR = Path("outputs")
GEN_DIR = OUTPUT_DIR / "generations"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
JUDGE_DIR = OUTPUT_DIR / "judge"

PROMPT_IDS_REQUESTED = list(range(27, 34))  # inclusive range 27..33
NUM_SAMPLES_PER_PROMPT = 8
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 1100

# Model list provided by user.
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")


@dataclass
class HttpResult:
    data: Dict[str, Any]
    status_code: int


class HttpClient:
    """Small HTTP client with retry logic for transient failures."""

    def __init__(self, timeout_seconds: int = 120, max_retries: int = 4) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def post_json(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> HttpResult:
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(
            url=url,
            headers=headers,
            data=payload,
            method="POST",
        )

        attempt = 0
        while True:
            attempt += 1
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read().decode("utf-8")
                    parsed = json.loads(raw)
                    return HttpResult(data=parsed, status_code=resp.status)
            except error.HTTPError as e:
                raw = e.read().decode("utf-8") if e.fp else ""
                retriable = e.code in {408, 409, 429, 500, 502, 503, 504}
                if attempt <= self.max_retries and retriable:
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                raise RuntimeError(f"HTTPError {e.code}: {raw}") from e
            except (error.URLError, TimeoutError) as e:
                if attempt <= self.max_retries:
                    time.sleep((2 ** (attempt - 1)) + random.random())
                    continue
                raise RuntimeError(f"Network error after retries: {e}") from e


def parse_openai_response_text(data: Dict[str, Any]) -> str:
    # Supports responses API structures.
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()

    output = data.get("output", [])
    texts: List[str] = []
    for item in output:
        for block in item.get("content", []):
            txt = block.get("text")
            if txt:
                texts.append(txt)
    if texts:
        return "\n".join(texts).strip()

    # Fallback for chat-completions-like structures.
    choices = data.get("choices", [])
    if choices:
        message = choices[0].get("message", {})
        text = message.get("content")
        if isinstance(text, str):
            return text.strip()

    return ""


def parse_anthropic_response_text(data: Dict[str, Any]) -> str:
    contents = data.get("content", [])
    texts: List[str] = []
    for item in contents:
        txt = item.get("text")
        if txt:
            texts.append(txt)
    return "\n".join(texts).strip()


def parse_gemini_response_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [p.get("text", "") for p in parts if p.get("text")]
    return "\n".join(texts).strip()


def parse_xai_response_text(data: Dict[str, Any]) -> str:
    choices = data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    return ""


class ModelClient:
    """Provider-specific generation requests via direct HTTPS."""

    def __init__(self, http: HttpClient) -> None:
        self.http = http

    def generate(
        self,
        model_spec: Dict[str, str],
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        provider = model_spec["provider"]
        model = model_spec["api_model"]
        if provider == "openai":
            return self._generate_openai(model_spec, prompt, temperature, top_p, max_tokens)
        if provider == "anthropic":
            return self._generate_anthropic(model_spec, prompt, temperature, max_tokens)
        if provider == "gemini":
            return self._generate_gemini(model, prompt, temperature, top_p, max_tokens)
        if provider == "xai":
            return self._generate_xai(model, prompt, temperature, top_p, max_tokens)
        raise ValueError(f"Unsupported provider: {provider}")

    def _generate_openai(
        self,
        model_spec: Dict[str, str],
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        model = model_spec["api_model"]
        is_reasoning = model_spec["thinking"] == "reasoning"
        body: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        # For GPT-5.2 pro variants, switch reasoning effort high vs low.
        body["reasoning"] = {"effort": "high" if is_reasoning else "low"}

        # Some OpenAI models (e.g., GPT-5 family) reject sampling params such as top_p.
        # Retry once by removing the specific unsupported parameter reported by the API.
        for _ in range(2):
            try:
                result = self.http.post_json(
                    url="https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    body=body,
                )
                break
            except RuntimeError as e:
                msg = str(e)
                match = re.search(r"Unsupported parameter: '([^']+)'", msg)
                if not match:
                    raise
                unsupported = match.group(1)
                if unsupported not in body:
                    raise
                del body[unsupported]
        else:
            raise RuntimeError("OpenAI request failed after removing unsupported parameters")

        text = parse_openai_response_text(result.data)
        return text, result.data

    def _generate_anthropic(
        self,
        model_spec: Dict[str, str],
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        model = model_spec["api_model"]
        thinking_enabled = model_spec["thinking"] == "reasoning"
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if thinking_enabled:
            # Enable Claude thinking for the reasoning variant.
            body["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        # Requested style: direct Request call against /v1/messages.
        result = self.http.post_json(
            url="https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            body=body,
        )
        text = parse_anthropic_response_text(result.data)
        return text, result.data

    def _generate_gemini(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "maxOutputTokens": max_tokens,
            },
        }
        result = self.http.post_json(
            url=url,
            headers={"Content-Type": "application/json"},
            body=body,
        )
        text = parse_gemini_response_text(result.data)
        return text, result.data

    def _generate_xai(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY not set")

        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        result = self.http.post_json(
            url="https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body=body,
        )
        text = parse_xai_response_text(result.data)
        return text, result.data


def load_prompt_items(dataset_path: Path, prompt_ids: List[int], max_modifiers: Optional[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    missing: List[int] = []
    rows: List[Dict[str, Any]] = []

    for prompt_id in prompt_ids:
        key = str(prompt_id)
        item = data.get(key)
        if item is None:
            missing.append(prompt_id)
            continue

        mods = item.get("seed_modifiers", [])
        if max_modifiers is not None:
            mods = mods[:max_modifiers]

        rows.append(
            {
                "prompt_id": prompt_id,
                "title": item.get("title"),
                "category": item.get("category"),
                "writing_prompt": item.get("writing_prompt", ""),
                "seed_modifiers": mods,
            }
        )

    return rows, missing


def build_final_prompt(writing_prompt: str, seed_modifier: str) -> str:
    # Explicitly appends the seed modifier to the end of the prompt text.
    return f"{writing_prompt.rstrip()}\n\nSeed modifier: {seed_modifier.strip()}"


def save_model_doc(out_path: Path, doc: Dict[str, Any]) -> None:
    """Persist one model's generation document as JSON."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def run_generation_task(
    client: ModelClient,
    model_spec: Dict[str, str],
    prompt_item: Dict[str, Any],
    modifier_idx: int,
    modifier: str,
    sample_idx: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Single generation request task used by the thread pool."""
    final_prompt = build_final_prompt(prompt_item["writing_prompt"], modifier)
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
        text, raw = client.generate(
            model_spec=model_spec,
            prompt=final_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        record["response_text"] = text
        record["raw_api_response"] = raw
    except Exception as e:
        record["error"] = str(e)

    record["latency_sec"] = round(time.time() - started, 3)
    return record


def generate_all(args: argparse.Namespace) -> None:
    load_dotenv()
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    prompt_items, missing = load_prompt_items(DATASET_PATH, PROMPT_IDS_REQUESTED, args.max_modifiers)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    http = HttpClient(timeout_seconds=args.timeout, max_retries=args.retries)
    client = ModelClient(http)

    for model_spec in MODEL_SPECS:
        display_name = model_spec["display_name"]
        out_path = GEN_DIR / f"{slugify(display_name)}.json"

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
                "num_samples_per_prompt": NUM_SAMPLES_PER_PROMPT,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            },
            "records": [],
        }
        # Create file early so progress is visible immediately.
        save_model_doc(out_path, doc)

        print(f"[generate] model={display_name} workers={args.max_workers}")
        futures: Dict[concurrent.futures.Future, Tuple[int, int, int]] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            for prompt_item in prompt_items:
                prompt_id = prompt_item["prompt_id"]
                mods = prompt_item["seed_modifiers"]
                if not mods:
                    print(f"  [warn] prompt_id={prompt_id} has no seed_modifiers")
                    continue

                for modifier_idx, modifier in enumerate(mods):
                    for sample_idx in range(NUM_SAMPLES_PER_PROMPT):
                        fut = executor.submit(
                            run_generation_task,
                            client,
                            model_spec,
                            prompt_item,
                            modifier_idx,
                            modifier,
                            sample_idx,
                            args.temperature,
                            args.top_p,
                            args.max_tokens,
                        )
                        futures[fut] = (prompt_id, modifier_idx, sample_idx)

            for fut in concurrent.futures.as_completed(futures):
                prompt_id, modifier_idx, sample_idx = futures[fut]
                record = fut.result()
                doc["records"].append(record)
                # Save after each completed request so partial results are never lost.
                save_model_doc(out_path, doc)
                print(
                    f"  prompt={prompt_id} mod={modifier_idx} sample={sample_idx + 1}/{NUM_SAMPLES_PER_PROMPT} "
                    f"status={'ok' if record['error'] is None else 'error'}"
                )

        # Keep records in a stable order for downstream analysis and diffs.
        doc["records"].sort(
            key=lambda r: (r["prompt_id"], r["seed_modifier_index"], r["sample_index"], r["created_at_utc"])
        )

        save_model_doc(out_path, doc)
        print(f"[saved] {out_path}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    cos_sim = float(np.dot(a, b) / denom)
    return 1.0 - cos_sim


def fetch_openai_embeddings(texts: List[str], model: str) -> List[List[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    http = HttpClient(timeout_seconds=120, max_retries=4)
    vectors: List[List[float]] = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        body = {
            "model": model,
            "input": batch,
        }
        result = http.post_json(
            url="https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body=body,
        )

        data = result.data.get("data", [])
        # Keep order stable by sorting index.
        data_sorted = sorted(data, key=lambda x: x.get("index", 0))
        vectors.extend([d["embedding"] for d in data_sorted])

    return vectors


def analyze_embeddings(args: argparse.Namespace) -> None:
    load_dotenv()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found under outputs/generations")

    report: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "embedding_model": args.embedding_model,
        "grouping": "model + prompt_id + seed_modifier_index",
        "groups": [],
        "by_model": {},
    }

    for model_file in model_files:
        with model_file.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        model_name = doc["model"]["display_name"]
        # Group records that are valid generations.
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in doc.get("records", []):
            if rec.get("error") is not None:
                continue
            txt = rec.get("response_text")
            if not txt:
                continue
            key = f"{rec['prompt_id']}::{rec['seed_modifier_index']}"
            grouped.setdefault(key, []).append(rec)

        model_group_means: List[float] = []
        for key, records in grouped.items():
            if len(records) < 2:
                continue

            texts = [r["response_text"] for r in records]
            embeds = fetch_openai_embeddings(texts, args.embedding_model)
            arrs = [np.array(v, dtype=np.float64) for v in embeds]

            pairs: List[Tuple[int, int, float]] = []
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    dist = cosine_distance(arrs[i], arrs[j])
                    pairs.append((i, j, dist))

            dists = [p[2] for p in pairs]
            if not dists:
                continue

            mean_dist = float(np.mean(dists))
            std_dist = float(np.std(dists))
            min_pair = min(pairs, key=lambda x: x[2])
            max_pair = max(pairs, key=lambda x: x[2])
            model_group_means.append(mean_dist)

            prompt_id, modifier_idx = key.split("::")
            report["groups"].append(
                {
                    "model": model_name,
                    "prompt_id": int(prompt_id),
                    "seed_modifier_index": int(modifier_idx),
                    "num_samples": len(records),
                    "pairwise_count": len(dists),
                    "mean_cosine_distance": mean_dist,
                    "std_cosine_distance": std_dist,
                    "min_cosine_distance": float(min(dists)),
                    "max_cosine_distance": float(max(dists)),
                    "least_diverse_pair": {
                        "sample_index_a": records[min_pair[0]]["sample_index"],
                        "sample_index_b": records[min_pair[1]]["sample_index"],
                        "distance": min_pair[2],
                    },
                    "most_diverse_pair": {
                        "sample_index_a": records[max_pair[0]]["sample_index"],
                        "sample_index_b": records[max_pair[1]]["sample_index"],
                        "distance": max_pair[2],
                    },
                }
            )

        report["by_model"][model_name] = {
            "num_groups": len(model_group_means),
            "mean_group_distance": float(np.mean(model_group_means)) if model_group_means else None,
            "std_group_distance": float(np.std(model_group_means)) if model_group_means else None,
            "max_group_distance": float(np.max(model_group_means)) if model_group_means else None,
            "min_group_distance": float(np.min(model_group_means)) if model_group_means else None,
        }

    # Rank models by average embedding-space diversity.
    sortable = [
        (name, vals["mean_group_distance"])
        for name, vals in report["by_model"].items()
        if vals["mean_group_distance"] is not None
    ]
    sortable.sort(key=lambda x: x[1], reverse=True)
    report["ranking_by_mean_group_distance"] = [
        {"model": name, "mean_group_distance": score} for name, score in sortable
    ]

    out_path = ANALYSIS_DIR / "embedding_diversity.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")


def build_judge_prompt(model_name: str, prompt_id: int, modifier_text: str, responses: List[Dict[str, Any]]) -> str:
    """Creates a judge prompt for choosing most/least diverse outputs among samples."""
    response_blocks = []
    for idx, rec in enumerate(responses, start=1):
        response_blocks.append(f"Response {idx}:\n{rec['response_text']}")

    joined = "\n\n".join(response_blocks)
    return (
        "You are evaluating creative-writing outputs for diversity across multiple generations of the same prompt.\n"
        "Task:\n"
        "1) Identify the TWO responses that are MOST diverse from each other.\n"
        "2) Identify the TWO responses that are LEAST diverse (most similar).\n"
        "3) Briefly justify each choice using style, structure, tone, narrative perspective, and content differences.\n"
        "4) Return strict JSON with keys: most_diverse_pair, least_diverse_pair, rationale.\n\n"
        f"Model under evaluation: {model_name}\n"
        f"Prompt ID: {prompt_id}\n"
        f"Seed modifier: {modifier_text}\n\n"
        f"{joined}"
    )


def create_judge_placeholders(_: argparse.Namespace) -> None:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found under outputs/generations")

    for model_file in model_files:
        with model_file.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        model_name = doc["model"]["display_name"]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in doc.get("records", []):
            if rec.get("error") is None and rec.get("response_text"):
                key = f"{rec['prompt_id']}::{rec['seed_modifier_index']}"
                grouped.setdefault(key, []).append(rec)

        judge_doc: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "model": model_name,
            "judge_config": {
                "judge_provider": "PLACEHOLDER_FROM_ENV",
                "judge_model": "PLACEHOLDER_FROM_ENV",
                "judge_api_key_env_var": "JUDGE_API_KEY",
                "status": "prompt_templates_only_no_api_call",
            },
            "items": [],
        }

        for key, records in grouped.items():
            if len(records) < 2:
                continue
            # Stable order by sample index so prompt mapping remains consistent.
            records = sorted(records, key=lambda r: r["sample_index"])
            prompt_id, modifier_idx = key.split("::")
            modifier_text = records[0]["seed_modifier"]
            judge_prompt = build_judge_prompt(model_name, int(prompt_id), modifier_text, records)
            judge_doc["items"].append(
                {
                    "prompt_id": int(prompt_id),
                    "seed_modifier_index": int(modifier_idx),
                    "seed_modifier": modifier_text,
                    "sample_record_ids": [r["record_id"] for r in records],
                    "judge_prompt": judge_prompt,
                    "judge_result": None,
                }
            )

        out_path = JUDGE_DIR / f"judge_placeholder_{slugify(model_name)}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(judge_doc, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM diversity benchmark pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="Run generation for all configured models")
    p_generate.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_generate.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p_generate.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_generate.add_argument("--max-modifiers", type=int, default=None, help="Optional cap for seed modifiers per prompt")
    p_generate.add_argument("--max-workers", type=int, default=8, help="Number of parallel generation workers per model")
    p_generate.add_argument("--timeout", type=int, default=120)
    p_generate.add_argument("--retries", type=int, default=4)
    p_generate.set_defaults(func=generate_all)

    p_embed = sub.add_parser("analyze-embeddings", help="Compute embedding-space diversity metrics")
    p_embed.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    p_embed.set_defaults(func=analyze_embeddings)

    p_judge = sub.add_parser("build-judge-placeholders", help="Create LLM-as-judge placeholder prompts")
    p_judge.set_defaults(func=create_judge_placeholders)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
