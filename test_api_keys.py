#!/usr/bin/env python3
"""Quick API key checker.

Default mode checks presence only.
Use --verify to run lightweight provider auth checks.
"""

import argparse
import json
import os
from typing import Dict, Tuple
from urllib import error, request

from dotenv import load_dotenv

REQUIRED_KEYS = {
    "OPENAI_API_KEY": "OpenAI",
    "ANTHROPIC_API_KEY": "Anthropic",
    "GEMINI_API_KEY": "Google Gemini",
    "XAI_API_KEY": "xAI",
}


def has_value(name: str) -> bool:
    v = os.getenv(name)
    return bool(v and v.strip())


def get(url: str, headers: Dict[str, str]) -> Tuple[bool, str]:
    req = request.Request(url=url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=25) as resp:
            _ = resp.read()
            return True, f"HTTP {resp.status}"
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        return False, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return False, str(e)


def verify_openai() -> Tuple[bool, str]:
    return get(
        "https://api.openai.com/v1/models",
        {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}"},
    )


def verify_anthropic() -> Tuple[bool, str]:
    # POST /v1/messages with minimal request and max_tokens=1.
    body = {
        "model": "claude-opus-4-6",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = request.Request(
        url="https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            _ = resp.read()
            return True, f"HTTP {resp.status}"
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        # 400 can mean model/params issue but key may still be valid.
        if e.code == 400 and "invalid x-api-key" not in body.lower():
            return True, f"HTTP 400 (key likely valid): {body[:200]}"
        return False, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return False, str(e)


def verify_gemini() -> Tuple[bool, str]:
    key = os.getenv("GEMINI_API_KEY", "")
    return get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
        {"Content-Type": "application/json"},
    )


def verify_xai() -> Tuple[bool, str]:
    return get(
        "https://api.x.ai/v1/models",
        {"Authorization": f"Bearer {os.getenv('XAI_API_KEY', '')}"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check API keys in .env")
    parser.add_argument("--verify", action="store_true", help="Also verify keys against provider endpoints")
    args = parser.parse_args()

    load_dotenv()

    print("Checking key presence:")
    missing = []
    for env_name, provider in REQUIRED_KEYS.items():
        ok = has_value(env_name)
        status = "OK" if ok else "MISSING"
        print(f"- {provider:14} {env_name:20} {status}")
        if not ok:
            missing.append(env_name)

    if missing:
        print("\nMissing keys found. Add them to .env before running model calls.")

    if not args.verify:
        return

    print("\nVerifying keys with provider endpoints:")
    checks = [
        ("OpenAI", "OPENAI_API_KEY", verify_openai),
        ("Anthropic", "ANTHROPIC_API_KEY", verify_anthropic),
        ("Gemini", "GEMINI_API_KEY", verify_gemini),
        ("xAI", "XAI_API_KEY", verify_xai),
    ]

    for provider, env_name, fn in checks:
        if not has_value(env_name):
            print(f"- {provider:14} SKIPPED (missing {env_name})")
            continue
        ok, msg = fn()
        print(f"- {provider:14} {'OK' if ok else 'FAIL'}  {msg}")


if __name__ == "__main__":
    main()
