#!/usr/bin/env python3
"""Diversity pipeline for Creative Writing Bench prompts.

This script supports:
1) Generation across multiple providers/models.
2) Embedding-space diversity analysis.
3) LLM-as-judge placeholder artifact generation.
"""

import argparse
import concurrent.futures
import hashlib
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
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer


DATASET_PATH = Path("creative_writing_prompts_v3.json")
OUTPUT_DIR = Path("outputs")
GEN_DIR = OUTPUT_DIR / "generations"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
JUDGE_DIR = OUTPUT_DIR / "judge"
EMBED_DIR = OUTPUT_DIR / "embeddings"
ACCURACY_DIR = OUTPUT_DIR / "accuracy"

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

_HF_EMBED_MODEL = None
_HF_EMBED_TOKENIZER = None
_HF_EMBED_MODEL_ID = None


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


def _load_hf_embedding_model(model_id: str) -> Tuple[Any, Any]:
    global _HF_EMBED_MODEL, _HF_EMBED_TOKENIZER, _HF_EMBED_MODEL_ID
    if _HF_EMBED_MODEL is not None and _HF_EMBED_TOKENIZER is not None and _HF_EMBED_MODEL_ID == model_id:
        return _HF_EMBED_TOKENIZER, _HF_EMBED_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        model = model.to("mps")
    else:
        model = model.to("cpu")
    model.eval()

    _HF_EMBED_TOKENIZER = tokenizer
    _HF_EMBED_MODEL = model
    _HF_EMBED_MODEL_ID = model_id
    return tokenizer, model


def fetch_hf_e5_embeddings(texts: List[str], model_id: str) -> List[List[float]]:
    tokenizer, model = _load_hf_embedding_model(model_id)
    prefixed = [f"query: {text}" for text in texts]
    vectors: List[List[float]] = []
    batch_size = 32

    for i in range(0, len(prefixed), batch_size):
        batch = prefixed[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.extend(pooled.detach().cpu().tolist())

    return vectors


def fetch_embeddings(texts: List[str], embedding_model: str) -> List[List[float]]:
    if embedding_model.startswith("intfloat/"):
        return fetch_hf_e5_embeddings(texts, embedding_model)
    return fetch_openai_embeddings(texts, embedding_model)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_embedding_cache(cache_path: Path, embedding_model: str) -> Dict[str, Any]:
    if not cache_path.exists():
        return {"embedding_model": embedding_model, "entries": {}}
    with cache_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("embedding_model") != embedding_model:
        # Model changed, so start a fresh cache for correctness.
        return {"embedding_model": embedding_model, "entries": {}}
    if "entries" not in doc:
        doc["entries"] = {}
    return doc


def save_embedding_cache(cache_path: Path, cache_doc: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache_doc, f, ensure_ascii=False, indent=2)


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def cosine_distance_to_centroid(vectors: List[np.ndarray]) -> List[float]:
    centroid = np.mean(np.stack(vectors, axis=0), axis=0)
    return [cosine_distance(v, centroid) for v in vectors]


def analyze_embeddings(args: argparse.Namespace) -> None:
    load_dotenv()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    if getattr(args, "generation_files", None):
        model_files = [Path(p) for p in args.generation_files]
    else:
        model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found under outputs/generations")
    print("[embed] generation files to analyze:")
    for model_file in model_files:
        print(f"[embed]   {model_file}")

    index_summary: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "embedding_model": args.embedding_model,
        "grouping": "prompt_id + seed_modifier_index",
        "per_model_files": [],
    }

    for model_file in model_files:
        print(f"[embed] loading model file: {model_file}")
        with model_file.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        model_name = doc["model"]["display_name"]
        model_slug = model_file.stem
        cache_path = EMBED_DIR / f"{model_slug}_embeddings.json"
        metrics_path = ANALYSIS_DIR / f"{model_slug}_embedding_metrics.json"
        cache_doc = load_embedding_cache(cache_path, args.embedding_model)

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

        model_report: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "source_generation_file": str(model_file),
            "embedding_cache_file": str(cache_path),
            "embedding_model": args.embedding_model,
            "model": doc.get("model", {}),
            "grouping": "prompt_id + seed_modifier_index",
            "groups": [],
            "summary": {},
        }
        avg_to_center_values: List[float] = []
        max_pair_values: List[float] = []
        group_items = sorted(grouped.items(), key=lambda x: x[0])
        total_groups = len(group_items)
        print(f"[embed] model={model_name} groups={total_groups}")

        for idx, (key, records) in enumerate(group_items, start=1):
            if len(records) < 2:
                continue

            # Stable ordering for reproducible pair ids and metrics.
            records = sorted(records, key=lambda r: r["sample_index"])

            # Fetch cached embeddings when possible; only call API for misses.
            pending_texts: List[str] = []
            pending_record_ids: List[str] = []
            vectors_by_record: Dict[str, List[float]] = {}
            for rec in records:
                rid = rec["record_id"]
                txt = rec["response_text"]
                th = text_hash(txt)
                cached = cache_doc["entries"].get(rid)
                if cached and cached.get("text_hash") == th:
                    vectors_by_record[rid] = cached["embedding"]
                else:
                    pending_record_ids.append(rid)
                    pending_texts.append(txt)

            if pending_texts:
                print(
                    f"[embed] model={model_name} group={idx}/{total_groups} "
                    f"fetch_embeddings={len(pending_texts)}"
                )
                new_vectors = fetch_embeddings(pending_texts, args.embedding_model)
                for rid, txt, vec in zip(pending_record_ids, pending_texts, new_vectors):
                    vectors_by_record[rid] = vec
                    cache_doc["entries"][rid] = {
                        "text_hash": text_hash(txt),
                        "embedding": vec,
                    }
                save_embedding_cache(cache_path, cache_doc)

            arrs = [np.array(vectors_by_record[r["record_id"]], dtype=np.float64) for r in records]

            pairs: List[Tuple[int, int, float]] = []
            for i in range(len(arrs)):
                for j in range(i + 1, len(arrs)):
                    dist = cosine_distance(arrs[i], arrs[j])
                    pairs.append((i, j, dist))

            dists = [p[2] for p in pairs]
            if not dists:
                continue

            max_pair = max(pairs, key=lambda x: x[2])
            center_dists = cosine_distance_to_centroid(arrs)
            avg_dist_to_center = float(np.mean(center_dists))
            max_pair_distance = float(max_pair[2])
            avg_to_center_values.append(avg_dist_to_center)
            max_pair_values.append(max_pair_distance)

            prompt_id, modifier_idx = key.split("::")
            model_report["groups"].append(
                {
                    "prompt_id": int(prompt_id),
                    "seed_modifier_index": int(modifier_idx),
                    "num_samples": len(records),
                    "distance_metric": "cosine_distance",
                    "max_pairwise_cosine_distance": max_pair_distance,
                    "average_cosine_distance_to_centroid": avg_dist_to_center,
                    "furthest_pair": {
                        "sample_index_a": records[max_pair[0]]["sample_index"],
                        "sample_index_b": records[max_pair[1]]["sample_index"],
                        "record_id_a": records[max_pair[0]]["record_id"],
                        "record_id_b": records[max_pair[1]]["record_id"],
                        "distance": max_pair_distance,
                    },
                    "sample_record_ids": [r["record_id"] for r in records],
                }
            )
            prompt_id, modifier_idx = key.split("::")
            print(
                f"[embed] model={model_name} group={idx}/{total_groups} "
                f"prompt={prompt_id} mod={modifier_idx} "
                f"ok avg_to_centroid={avg_dist_to_center:.4f} max_pair={max_pair_distance:.4f}"
            )
            save_json(metrics_path, model_report)

        model_report["summary"] = {
            "model_name": model_name,
            "num_groups": len(model_report["groups"]),
            "mean_of_max_pairwise_distance": float(np.mean(max_pair_values)) if max_pair_values else None,
            "mean_of_avg_distance_to_centroid": float(np.mean(avg_to_center_values)) if avg_to_center_values else None,
        }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(model_report, f, ensure_ascii=False, indent=2)
        print(f"[saved] {metrics_path}")
        print(f"[saved] {cache_path}")

        index_summary["per_model_files"].append(
            {
                "model_name": model_name,
                "source_generation_file": str(model_file),
                "metrics_file": str(metrics_path),
                "embedding_cache_file": str(cache_path),
            }
        )

    out_path = ANALYSIS_DIR / "embedding_metrics_index.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(index_summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")


def analyze_baseline_deviation(args: argparse.Namespace) -> None:
    """Compute baseline deviation metrics using cosine distance to centroid.

    Outputs (per model):
    - *_baseline_deviation_metrics.json
    - *_baseline_deviation_details.json
    """
    load_dotenv()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    if getattr(args, "generation_files", None):
        model_files = [Path(p) for p in args.generation_files]
    else:
        model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found under outputs/generations")
    print("[baseline] generation files to analyze:")
    for model_file in model_files:
        print(f"[baseline]   {model_file}")

    index_doc: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "embedding_model": args.embedding_model,
        "metric_definition": {
            "average_deviation": "mean cosine distance from each sample embedding to group centroid",
            "max_deviation": "max cosine distance from any sample embedding to group centroid",
        },
        "per_model_files": [],
    }

    for model_file in model_files:
        print(f"[baseline] loading model file: {model_file}")
        try:
            with model_file.open("r", encoding="utf-8") as f:
                gen_doc = json.load(f)
        except Exception as e:
            print(f"[baseline][error] failed to load {model_file}: {e}")
            index_doc["per_model_files"].append(
                {
                    "source_generation_file": str(model_file),
                    "status": "error_loading_generation_file",
                    "error": str(e),
                }
            )
            save_json(ANALYSIS_DIR / "baseline_deviation_index.json", index_doc)
            continue

        model_name = gen_doc["model"]["display_name"]
        model_slug = model_file.stem
        cache_path = EMBED_DIR / f"{model_slug}_embeddings.json"
        metrics_path = ANALYSIS_DIR / f"{model_slug}_baseline_deviation_metrics.json"
        details_path = ANALYSIS_DIR / f"{model_slug}_baseline_deviation_details.json"

        cache_doc = load_embedding_cache(cache_path, args.embedding_model)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in gen_doc.get("records", []):
            if rec.get("error") is not None:
                continue
            if not rec.get("response_text"):
                continue
            key = f"{rec['prompt_id']}::{rec['seed_modifier_index']}"
            grouped.setdefault(key, []).append(rec)

        metrics_doc: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "source_generation_file": str(model_file),
            "embedding_cache_file": str(cache_path),
            "embedding_model": args.embedding_model,
            "model": gen_doc.get("model", {}),
            "groups": [],
            "group_errors": [],
            "summary": {},
        }
        details_doc: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "source_generation_file": str(model_file),
            "embedding_cache_file": str(cache_path),
            "embedding_model": args.embedding_model,
            "model": gen_doc.get("model", {}),
            "groups": [],
            "group_errors": [],
        }

        avg_values: List[float] = []
        max_values: List[float] = []
        group_items = sorted(grouped.items(), key=lambda x: x[0])
        total_groups = len(group_items)
        print(f"[baseline] model={model_name} groups={total_groups}")

        for idx, (key, records) in enumerate(group_items, start=1):
            if len(records) < 2:
                continue
            try:
                records = sorted(records, key=lambda r: r["sample_index"])
                pending_texts: List[str] = []
                pending_record_ids: List[str] = []
                vectors_by_record: Dict[str, List[float]] = {}

                for rec in records:
                    rid = rec["record_id"]
                    txt = rec["response_text"]
                    th = text_hash(txt)
                    cached = cache_doc["entries"].get(rid)
                    if cached and cached.get("text_hash") == th:
                        vectors_by_record[rid] = cached["embedding"]
                    else:
                        pending_record_ids.append(rid)
                        pending_texts.append(txt)

                if pending_texts:
                    print(
                        f"[baseline] model={model_name} group={idx}/{total_groups} "
                        f"fetch_embeddings={len(pending_texts)}"
                    )
                    new_vectors = fetch_embeddings(pending_texts, args.embedding_model)
                    for rid, txt, vec in zip(pending_record_ids, pending_texts, new_vectors):
                        vectors_by_record[rid] = vec
                        cache_doc["entries"][rid] = {
                            "text_hash": text_hash(txt),
                            "embedding": vec,
                        }
                    save_embedding_cache(cache_path, cache_doc)

                arrs = [np.array(vectors_by_record[r["record_id"]], dtype=np.float64) for r in records]
                centroid = np.mean(np.stack(arrs, axis=0), axis=0)
                deviations = [cosine_distance(v, centroid) for v in arrs]
                avg_dev = float(np.mean(deviations))
                max_dev = float(np.max(deviations))
                max_i = int(np.argmax(deviations))

                avg_values.append(avg_dev)
                max_values.append(max_dev)

                prompt_id, modifier_idx = key.split("::")
                metrics_doc["groups"].append(
                    {
                        "prompt_id": int(prompt_id),
                        "seed_modifier_index": int(modifier_idx),
                        "num_samples": len(records),
                        "average_cosine_deviation_to_centroid": avg_dev,
                        "max_cosine_deviation_to_centroid": max_dev,
                        "max_deviation_record_id": records[max_i]["record_id"],
                        "max_deviation_sample_index": records[max_i]["sample_index"],
                        "sample_record_ids": [r["record_id"] for r in records],
                    }
                )

                details_doc["groups"].append(
                    {
                        "prompt_id": int(prompt_id),
                        "seed_modifier_index": int(modifier_idx),
                        "num_samples": len(records),
                        "centroid_norm": float(np.linalg.norm(centroid)),
                        "samples": [
                            {
                                "record_id": rec["record_id"],
                                "sample_index": rec["sample_index"],
                                "cosine_deviation_to_centroid": float(dev),
                            }
                            for rec, dev in zip(records, deviations)
                        ],
                    }
                )

                print(
                    f"[baseline] model={model_name} group={idx}/{total_groups} "
                    f"ok avg={avg_dev:.4f} max={max_dev:.4f}"
                )
            except Exception as e:
                prompt_id, modifier_idx = key.split("::")
                err_obj = {
                    "prompt_id": int(prompt_id),
                    "seed_modifier_index": int(modifier_idx),
                    "error": str(e),
                }
                metrics_doc["group_errors"].append(err_obj)
                details_doc["group_errors"].append(err_obj)
                print(
                    f"[baseline][error] model={model_name} group={idx}/{total_groups} "
                    f"prompt={prompt_id} mod={modifier_idx} error={e}"
                )
            finally:
                save_json(metrics_path, metrics_doc)
                save_json(details_path, details_doc)

        metrics_doc["summary"] = {
            "model_name": model_name,
            "num_groups": len(metrics_doc["groups"]),
            "num_group_errors": len(metrics_doc["group_errors"]),
            "mean_average_cosine_deviation_to_centroid": float(np.mean(avg_values)) if avg_values else None,
            "mean_max_cosine_deviation_to_centroid": float(np.mean(max_values)) if max_values else None,
        }
        save_json(metrics_path, metrics_doc)
        save_json(details_path, details_doc)
        print(f"[saved] {metrics_path}")
        print(f"[saved] {details_path}")
        print(f"[saved] {cache_path}")

        index_doc["per_model_files"].append(
            {
                "model_name": model_name,
                "source_generation_file": str(model_file),
                "metrics_file": str(metrics_path),
                "details_file": str(details_path),
                "embedding_cache_file": str(cache_path),
                "status": "ok",
                "num_groups": len(metrics_doc["groups"]),
                "num_group_errors": len(metrics_doc["group_errors"]),
            }
        )
        save_json(ANALYSIS_DIR / "baseline_deviation_index.json", index_doc)

    out_path = ANALYSIS_DIR / "baseline_deviation_index.json"
    save_json(out_path, index_doc)
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


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from model text output."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def build_accuracy_judge_prompt(prompt_text: str, response_text: str) -> str:
    return (
        "You are grading how well a response follows a writing prompt.\n"
        "Score from 0 to 100 where:\n"
        "- 0 = completely unrelated / fails prompt\n"
        "- 50 = partially follows prompt but misses important constraints\n"
        "- 100 = fully addresses prompt intent and constraints very well\n\n"
        "Return STRICT JSON only with keys:\n"
        "{ \"score\": <integer 0-100>, \"rationale\": \"<1-3 sentences>\" }\n\n"
        "Prompt:\n"
        f"{prompt_text}\n\n"
        "Response:\n"
        f"{response_text}"
    )


def score_response_with_gemini(
    http: HttpClient,
    judge_model: str,
    judge_temperature: float,
    prompt_text: str,
    response_text: str,
) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    judge_prompt = build_accuracy_judge_prompt(prompt_text, response_text)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}:generateContent?key={api_key}"
    body: Dict[str, Any] = {
        "contents": [{"parts": [{"text": judge_prompt}]}],
        "generationConfig": {
            "temperature": judge_temperature,
            "maxOutputTokens": 300,
            "responseMimeType": "application/json",
        },
    }
    result = http.post_json(
        url=url,
        headers={"Content-Type": "application/json"},
        body=body,
    )
    raw_text = parse_gemini_response_text(result.data)
    parsed = extract_first_json_obj(raw_text)
    if not parsed:
        raise RuntimeError(f"Failed to parse judge JSON: {raw_text[:500]}")

    score = parsed.get("score")
    if not isinstance(score, (int, float)):
        raise RuntimeError(f"Judge score missing/invalid: {parsed}")
    score_int = int(round(float(score)))
    score_int = max(0, min(100, score_int))
    rationale = parsed.get("rationale")
    if rationale is None:
        rationale = ""

    return {
        "score": score_int,
        "rationale": str(rationale),
        "judge_raw_text": raw_text,
    }


def analyze_accuracy_judge(args: argparse.Namespace) -> None:
    """LLM-as-a-judge accuracy scoring using Gemini judge model."""
    load_dotenv()
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    model_files = sorted(GEN_DIR.glob("*.json"))
    if not model_files:
        raise RuntimeError("No generation files found under outputs/generations")

    http = HttpClient(timeout_seconds=args.timeout, max_retries=args.retries)
    index_doc: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "judge_provider": "gemini",
        "judge_model": args.judge_model,
        "score_range": [0, 100],
        "per_model_files": [],
    }

    for model_file in model_files:
        print(f"[accuracy] loading model file: {model_file}")
        try:
            with model_file.open("r", encoding="utf-8") as f:
                gen_doc = json.load(f)
        except Exception as e:
            print(f"[accuracy][error] failed to load {model_file}: {e}")
            index_doc["per_model_files"].append(
                {
                    "source_generation_file": str(model_file),
                    "status": "error_loading_generation_file",
                    "error": str(e),
                }
            )
            save_json(ACCURACY_DIR / "accuracy_judge_index.json", index_doc)
            continue

        model_name = gen_doc.get("model", {}).get("display_name", model_file.stem)
        out_path = ACCURACY_DIR / f"{model_file.stem}_accuracy_judge.json"

        out_doc: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "source_generation_file": str(model_file),
            "judge_config": {
                "provider": "gemini",
                "model": args.judge_model,
                "temperature": args.judge_temperature,
                "score_range": [0, 100],
            },
            "model": gen_doc.get("model", {}),
            "groups": [],
            "group_errors": [],
            "summary": {},
        }
        save_json(out_path, out_doc)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in gen_doc.get("records", []):
            if rec.get("error") is not None:
                continue
            if not rec.get("response_text"):
                continue
            key = f"{rec['prompt_id']}::{rec['seed_modifier_index']}"
            grouped.setdefault(key, []).append(rec)

        all_scores: List[int] = []
        group_items = sorted(grouped.items(), key=lambda x: x[0])
        total_groups = len(group_items)
        print(f"[accuracy] model={model_name} groups={total_groups}")

        for idx, (key, records) in enumerate(group_items, start=1):
            records = sorted(records, key=lambda r: r["sample_index"])
            prompt_id, modifier_idx = key.split("::")

            group_obj: Dict[str, Any] = {
                "prompt_id": int(prompt_id),
                "seed_modifier_index": int(modifier_idx),
                "seed_modifier": records[0].get("seed_modifier"),
                "prompt_text": records[0].get("final_prompt"),
                "num_scored_responses": 0,
                "scores": [],
            }

            for rec in records:
                try:
                    judge = score_response_with_gemini(
                        http=http,
                        judge_model=args.judge_model,
                        judge_temperature=args.judge_temperature,
                        prompt_text=rec.get("final_prompt", ""),
                        response_text=rec.get("response_text", ""),
                    )
                    score_val = judge["score"]
                    all_scores.append(score_val)
                    group_obj["scores"].append(
                        {
                            "record_id": rec.get("record_id"),
                            "sample_index": rec.get("sample_index"),
                            "score": score_val,
                            "rationale": judge.get("rationale", ""),
                            "judge_raw_text": judge.get("judge_raw_text", ""),
                            "error": None,
                        }
                    )
                except Exception as e:
                    group_obj["scores"].append(
                        {
                            "record_id": rec.get("record_id"),
                            "sample_index": rec.get("sample_index"),
                            "score": None,
                            "rationale": "",
                            "judge_raw_text": "",
                            "error": str(e),
                        }
                    )
                    print(
                        f"[accuracy][error] model={model_name} group={idx}/{total_groups} "
                        f"prompt={prompt_id} mod={modifier_idx} sample={rec.get('sample_index')} error={e}"
                    )
                finally:
                    group_obj["num_scored_responses"] = sum(1 for s in group_obj["scores"] if s["score"] is not None)
                    # Save incrementally as requested.
                    existing = [g for g in out_doc["groups"] if not (g["prompt_id"] == int(prompt_id) and g["seed_modifier_index"] == int(modifier_idx))]
                    out_doc["groups"] = existing + [group_obj]
                    save_json(out_path, out_doc)

            print(
                f"[accuracy] model={model_name} group={idx}/{total_groups} "
                f"scored={group_obj['num_scored_responses']}/{len(records)}"
            )

        out_doc["summary"] = {
            "model_name": model_name,
            "num_groups": len(out_doc["groups"]),
            "total_scored_responses": len(all_scores),
            "mean_score": float(np.mean(all_scores)) if all_scores else None,
            "min_score": int(min(all_scores)) if all_scores else None,
            "max_score": int(max(all_scores)) if all_scores else None,
        }
        save_json(out_path, out_doc)
        print(f"[saved] {out_path}")

        index_doc["per_model_files"].append(
            {
                "model_name": model_name,
                "source_generation_file": str(model_file),
                "accuracy_file": str(out_path),
                "status": "ok",
                "num_groups": out_doc["summary"]["num_groups"],
                "total_scored_responses": out_doc["summary"]["total_scored_responses"],
                "mean_score": out_doc["summary"]["mean_score"],
            }
        )
        save_json(ACCURACY_DIR / "accuracy_judge_index.json", index_doc)

    save_json(ACCURACY_DIR / "accuracy_judge_index.json", index_doc)
    print(f"[saved] {ACCURACY_DIR / 'accuracy_judge_index.json'}")


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
    p_embed.add_argument("--embedding-model", type=str, default="intfloat/e5-large")
    p_embed.add_argument("--generation-files", nargs="*", default=None)
    p_embed.set_defaults(func=analyze_embeddings)

    p_dev = sub.add_parser(
        "analyze-baseline-deviation",
        help="Compute baseline deviation metrics (cosine distance to centroid) with per-model details",
    )
    p_dev.add_argument("--embedding-model", type=str, default="intfloat/e5-large")
    p_dev.add_argument("--generation-files", nargs="*", default=None)
    p_dev.set_defaults(func=analyze_baseline_deviation)

    p_acc = sub.add_parser(
        "analyze-accuracy-judge",
        help="Score prompt adherence accuracy (0-100) with Gemini judge; one output JSON per model",
    )
    p_acc.add_argument("--judge-model", type=str, default="gemini-3-flash-preview")
    p_acc.add_argument("--judge-temperature", type=float, default=0.0)
    p_acc.add_argument("--timeout", type=int, default=120)
    p_acc.add_argument("--retries", type=int, default=4)
    p_acc.set_defaults(func=analyze_accuracy_judge)

    p_judge = sub.add_parser("build-judge-placeholders", help="Create LLM-as-judge placeholder prompts")
    p_judge.set_defaults(func=create_judge_placeholders)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
