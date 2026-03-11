#!/usr/bin/env python3
"""LLM-as-judge accuracy scoring.

Key behavior:
- Scores prompt adherence 0-100
- One output JSON per model in outputs/accuracy/
- Incremental checkpoint saves
- Continue on errors
- Prompt-major scheduling: evaluate prompt groups across all models in lockstep
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request

from dotenv import load_dotenv

GEN_DIR = Path("outputs/generations")
ACCURACY_DIR = Path("outputs/accuracy")


class JudgeParseError(Exception):
    def __init__(self, message: str, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, doc: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


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
    return content.strip() if isinstance(content, str) else ""


def parse_anthropic_response_text(data: Dict[str, Any]) -> str:
    content = data.get("content", [])
    texts: List[str] = []
    for item in content:
        txt = item.get("text")
        if txt:
            texts.append(txt)
    return "\n".join(texts).strip()


def extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
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


class HttpClient:
    def __init__(self, timeout_seconds: int = 120, max_retries: int = 4) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def post_json(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        req = request.Request(url=url, headers=headers, data=payload, method="POST")

        attempt = 0
        while True:
            attempt += 1
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except error.HTTPError as e:
                raw = e.read().decode("utf-8", errors="ignore") if e.fp else ""
                retriable = e.code in {408, 409, 429, 500, 502, 503, 504}
                if attempt <= self.max_retries and retriable:
                    time.sleep(min(60, (2 ** (attempt - 1)) + 0.5))
                    continue
                raise RuntimeError(f"HTTPError {e.code}: {raw[:800]}") from e
            except Exception as e:
                if attempt <= self.max_retries:
                    time.sleep(min(60, (2 ** (attempt - 1)) + 0.5))
                    continue
                raise RuntimeError(f"Network error: {e}") from e


class RateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self.interval = 60.0 / requests_per_minute
        self.next_allowed = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        if now < self.next_allowed:
            time.sleep(self.next_allowed - now)
        self.next_allowed = time.monotonic() + self.interval


def build_group_prompt(prompt_text: str, responses: List[Dict[str, Any]]) -> str:
    blocks = []
    for rec in responses:
        blocks.append(
            f"sample_index: {rec['sample_index']}\n"
            f"record_id: {rec['record_id']}\n"
            f"response:\n{rec['response_text']}"
        )
    joined = "\n\n".join(blocks)
    return (
        "You are grading prompt adherence for creative-writing outputs.\n"
        "For EACH response, assign a score from 0 to 100 for how well it answers the prompt and follows constraints.\n"
        "Scale: 0=unrelated/fails prompt, 50=partially follows prompt, 100=fully addresses prompt constraints.\n\n"
        "Return STRICT JSON ONLY with this exact schema:\n"
        "{\n"
        "  \"scores\": [\n"
        "    {\"sample_index\": <int>, \"record_id\": \"<string>\", \"score\": <int 0-100>}\n"
        "  ]\n"
        "}\n"
        "No markdown. No extra keys.\n\n"
        "Prompt:\n"
        f"{prompt_text}\n\n"
        "Responses:\n"
        f"{joined}"
    )


def call_gemini_judge(
    http: HttpClient,
    limiter: RateLimiter,
    judge_model: str,
    judge_temperature: float,
    prompt_text: str,
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{judge_model}:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": build_group_prompt(prompt_text, responses)}]}],
        "generationConfig": {
            "temperature": judge_temperature,
            "maxOutputTokens": 900,
            "responseMimeType": "application/json",
        },
    }

    limiter.wait()
    data = http.post_json(url=url, headers={"Content-Type": "application/json"}, body=body)
    raw_text = parse_gemini_response_text(data)
    parsed = extract_first_json_obj(raw_text)
    if not parsed:
        raise JudgeParseError("Could not parse JSON from judge output", raw_text)
    if not isinstance(parsed.get("scores"), list):
        raise JudgeParseError("Judge JSON missing 'scores' list", raw_text)
    parsed["_judge_raw_text"] = raw_text
    return parsed


def call_xai_judge(
    http: HttpClient,
    limiter: RateLimiter,
    judge_model: str,
    judge_temperature: float,
    prompt_text: str,
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY not set")

    body = {
        "model": judge_model,
        "messages": [{"role": "user", "content": build_group_prompt(prompt_text, responses)}],
        "temperature": judge_temperature,
        "max_tokens": 900,
    }
    limiter.wait()
    data = http.post_json(
        url="https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        body=body,
    )
    raw_text = parse_xai_response_text(data)
    parsed = extract_first_json_obj(raw_text)
    if not parsed:
        raise JudgeParseError("Could not parse JSON from judge output", raw_text)
    if not isinstance(parsed.get("scores"), list):
        raise JudgeParseError("Judge JSON missing 'scores' list", raw_text)
    parsed["_judge_raw_text"] = raw_text
    return parsed


def call_anthropic_judge(
    http: HttpClient,
    limiter: RateLimiter,
    judge_model: str,
    judge_temperature: float,
    prompt_text: str,
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    body = {
        "model": judge_model,
        "max_tokens": 900,
        "temperature": judge_temperature,
        "messages": [{"role": "user", "content": build_group_prompt(prompt_text, responses)}],
    }
    limiter.wait()
    data = http.post_json(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        body=body,
    )
    raw_text = parse_anthropic_response_text(data)
    parsed = extract_first_json_obj(raw_text)
    if not parsed:
        raise JudgeParseError("Could not parse JSON from judge output", raw_text)
    if not isinstance(parsed.get("scores"), list):
        raise JudgeParseError("Judge JSON missing 'scores' list", raw_text)
    parsed["_judge_raw_text"] = raw_text
    return parsed


def call_judge(
    judge_provider: str,
    http: HttpClient,
    limiter: RateLimiter,
    judge_model: str,
    judge_temperature: float,
    prompt_text: str,
    responses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    provider = judge_provider.lower()
    if provider == "gemini":
        return call_gemini_judge(
            http=http,
            limiter=limiter,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            prompt_text=prompt_text,
            responses=responses,
        )
    if provider == "xai":
        return call_xai_judge(
            http=http,
            limiter=limiter,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            prompt_text=prompt_text,
            responses=responses,
        )
    if provider == "anthropic":
        return call_anthropic_judge(
            http=http,
            limiter=limiter,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            prompt_text=prompt_text,
            responses=responses,
        )
    raise ValueError(f"Unsupported judge provider: {judge_provider}")


def group_key(rec: Dict[str, Any]) -> str:
    return f"{rec['prompt_id']}::{rec['seed_modifier_index']}"


def key_sort_tuple(key: str) -> Tuple[int, int]:
    p, m = key.split("::")
    return int(p), int(m)


def load_contexts(generation_glob: str) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for model_file in sorted(GEN_DIR.glob(generation_glob)):
        try:
            with model_file.open("r", encoding="utf-8") as f:
                gen_doc = json.load(f)
        except Exception:
            continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in gen_doc.get("records", []):
            if rec.get("error") is not None:
                continue
            if not rec.get("response_text"):
                continue
            grouped.setdefault(group_key(rec), []).append(rec)

        contexts.append(
            {
                "model_file": model_file,
                "model_name": gen_doc.get("model", {}).get("display_name", model_file.stem),
                "model_meta": gen_doc.get("model", {}),
                "grouped": grouped,
            }
        )
    return contexts


def estimate_total_calls(contexts: List[Dict[str, Any]], batch_size: int) -> int:
    total = 0
    for ctx in contexts:
        for rows in ctx["grouped"].values():
            total += (len(rows) + batch_size - 1) // batch_size
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-major LLM accuracy judge")
    parser.add_argument("--judge-provider", type=str, default="gemini", choices=["gemini", "xai", "anthropic"])
    parser.add_argument("--judge-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rpm", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--generation-glob", type=str, default="*.json")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

    contexts = load_contexts(args.generation_glob)
    total_calls = estimate_total_calls(contexts, args.batch_size)
    est_minutes = total_calls / args.rpm if args.rpm > 0 else 0.0
    print(
        f"[estimate] model_files={len(contexts)} requests={total_calls} batch_size={args.batch_size} "
        f"rpm={args.rpm} est_min={est_minutes:.1f} (~{est_minutes/60:.2f}h)"
    )
    if args.estimate_only:
        return

    http = HttpClient(timeout_seconds=args.timeout, max_retries=args.retries)
    limiter = RateLimiter(requests_per_minute=args.rpm)

    index_doc: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "score_range": [0, 100],
        "schedule": "prompt-major",
        "batch_size": args.batch_size,
        "rate_limit_rpm": args.rpm,
        "per_model_files": [],
    }

    # Initialize per-model docs/files once.
    out_docs: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        model_file: Path = ctx["model_file"]
        out_path = ACCURACY_DIR / f"{model_file.stem}_accuracy_judge_batched.json"
        out_doc: Dict[str, Any] = {
            "created_at_utc": utc_now(),
            "source_generation_file": str(model_file),
            "judge_config": {
                "provider": args.judge_provider,
                "model": args.judge_model,
                "temperature": args.judge_temperature,
                "score_range": [0, 100],
                "rate_limit_rpm": args.rpm,
                "batch_size": args.batch_size,
                "schedule": "prompt-major",
            },
            "model": ctx["model_meta"],
            "groups": [],
            "group_errors": [],
            "summary": {},
        }
        out_docs[str(model_file)] = {"path": out_path, "doc": out_doc, "all_scores": []}
        save_json(out_path, out_doc)

    # Build global prompt-major group order.
    all_keys = sorted({k for ctx in contexts for k in ctx["grouped"].keys()}, key=key_sort_tuple)

    global_done = 0
    started = time.time()

    for key in all_keys:
        p, m = key_sort_tuple(key)
        print(f"[prompt-major] prompt={p} modifier={m}")

        for ctx in contexts:
            grouped = ctx["grouped"]
            if key not in grouped:
                continue

            records = sorted(grouped[key], key=lambda r: int(r.get("sample_index", 0)))
            model_file = str(ctx["model_file"])
            model_name = ctx["model_name"]
            state = out_docs[model_file]
            out_path: Path = state["path"]
            out_doc: Dict[str, Any] = state["doc"]
            all_scores: List[int] = state["all_scores"]

            group_obj: Dict[str, Any] = {
                "prompt_id": p,
                "seed_modifier_index": m,
                "seed_modifier": records[0].get("seed_modifier"),
                "prompt_text": records[0].get("final_prompt", ""),
                "num_input_responses": len(records),
                "num_scored_responses": 0,
                "scores": [],
                "judge_raw_text": [],
            }

            try:
                by_record_id = {r["record_id"]: r for r in records}
                by_sample_idx = {int(r.get("sample_index", -1)): r for r in records}

                for start_idx in range(0, len(records), args.batch_size):
                    batch = records[start_idx : start_idx + args.batch_size]
                    judged = call_judge(
                        judge_provider=args.judge_provider,
                        http=http,
                        limiter=limiter,
                        judge_model=args.judge_model,
                        judge_temperature=args.judge_temperature,
                        prompt_text=records[0].get("final_prompt", ""),
                        responses=batch,
                    )
                    group_obj["judge_raw_text"].append(judged.get("_judge_raw_text", ""))

                    for item in judged.get("scores", []):
                        rec = None
                        rid = item.get("record_id")
                        if isinstance(rid, str) and rid in by_record_id:
                            rec = by_record_id[rid]
                        else:
                            sidx = item.get("sample_index")
                            if isinstance(sidx, (int, float)):
                                rec = by_sample_idx.get(int(sidx))
                        if rec is None:
                            continue

                        score_raw = item.get("score")
                        if not isinstance(score_raw, (int, float)):
                            continue
                        score = max(0, min(100, int(round(float(score_raw)))))
                        all_scores.append(score)
                        group_obj["scores"].append(
                            {
                                "record_id": rec.get("record_id"),
                                "sample_index": rec.get("sample_index"),
                                "score": score,
                                "error": None,
                            }
                        )

                seen = {s["record_id"] for s in group_obj["scores"]}
                for rec in records:
                    if rec["record_id"] not in seen:
                        group_obj["scores"].append(
                            {
                                "record_id": rec.get("record_id"),
                                "sample_index": rec.get("sample_index"),
                                "score": None,
                                "error": "Judge output missing score for this response",
                            }
                        )

            except Exception as e:
                if isinstance(e, JudgeParseError):
                    group_obj["judge_raw_text"].append(e.raw_text)
                group_obj["scores"] = [
                    {
                        "record_id": rec.get("record_id"),
                        "sample_index": rec.get("sample_index"),
                        "score": None,
                        "error": str(e),
                    }
                    for rec in records
                ]
                out_doc["group_errors"].append(
                    {
                        "prompt_id": p,
                        "seed_modifier_index": m,
                        "error": str(e),
                    }
                )

            group_obj["num_scored_responses"] = sum(1 for s in group_obj["scores"] if s["score"] is not None)
            out_doc["groups"].append(group_obj)
            save_json(out_path, out_doc)

            # progress: calls consumed by this group
            consumed_calls = (len(records) + args.batch_size - 1) // args.batch_size
            global_done += consumed_calls
            elapsed_min = (time.time() - started) / 60.0
            remaining = max(total_calls - global_done, 0)
            eta_min = remaining / args.rpm if args.rpm > 0 else 0.0
            print(
                f"[accuracy] model={model_name} prompt={p} mod={m} scored={group_obj['num_scored_responses']}/{len(records)} "
                f"calls+={consumed_calls} global={global_done}/{total_calls} elapsed_min={elapsed_min:.1f} eta_min~{eta_min:.1f}"
            )

    # Finalize model summaries + index.
    for ctx in contexts:
        model_file = str(ctx["model_file"])
        state = out_docs[model_file]
        out_doc = state["doc"]
        all_scores = state["all_scores"]
        out_doc["summary"] = {
            "model_name": ctx["model_name"],
            "num_groups": len(out_doc["groups"]),
            "num_group_errors": len(out_doc["group_errors"]),
            "total_scored_responses": len(all_scores),
            "mean_score": (sum(all_scores) / len(all_scores)) if all_scores else None,
            "min_score": min(all_scores) if all_scores else None,
            "max_score": max(all_scores) if all_scores else None,
        }
        save_json(state["path"], out_doc)

        index_doc["per_model_files"].append(
            {
                "model_name": ctx["model_name"],
                "source_generation_file": str(ctx["model_file"]),
                "accuracy_file": str(state["path"]),
                "status": "ok",
                "num_groups": out_doc["summary"]["num_groups"],
                "num_group_errors": out_doc["summary"]["num_group_errors"],
                "total_scored_responses": out_doc["summary"]["total_scored_responses"],
                "mean_score": out_doc["summary"]["mean_score"],
            }
        )

    save_json(ACCURACY_DIR / "accuracy_judge_batched_index.json", index_doc)
    print(f"[saved] {ACCURACY_DIR / 'accuracy_judge_batched_index.json'}")


if __name__ == "__main__":
    main()
