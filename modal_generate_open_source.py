#!/usr/bin/env python3
"""Run open-source baseline generation on Modal with persistent volumes.

This reuses generate_open_source.py so local and Modal runs write the same JSON
schema and share the same resume/checkpoint behavior.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "project-lmdiversity-generation"
PROJECT_ROOT = Path("/root/project")
OUTPUTS_DIR = Path("/vol/outputs")
HF_CACHE_DIR = Path("/vol/hf-cache")
REPO_SYNC_DIR = Path("/vol/repo")

app = modal.App(APP_NAME)
outputs_volume = modal.Volume.from_name("lmdiversity-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("lmdiversity-hf-cache", create_if_missing=True)
repo_volume = modal.Volume.from_name("lmdiversity-repo", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=str(PROJECT_ROOT), ignore=[".git", "__pycache__", ".env"])
)


def sync_private_repo_if_configured() -> Path:
    """Optionally clone/pull a private repo into a persistent volume.

    Expected env vars if you want this path:
    - GITHUB_TOKEN
    - GITHUB_REPO_URL  (for example: github.com/user/repo.git)

    If unset, the baked-in local source tree is used.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    repo_url = os.environ.get("GITHUB_REPO_URL")
    if not github_token or not repo_url:
        return PROJECT_ROOT

    authenticated_url = f"https://x-access-token:{github_token}@{repo_url.removeprefix('https://')}"
    checkout_dir = REPO_SYNC_DIR / "repo"
    if not checkout_dir.exists():
        subprocess.run(["git", "clone", authenticated_url, str(checkout_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(checkout_dir), "pull", "--ff-only"], check=True)
    return checkout_dir


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60 * 10,
    volumes={
        "/vol/outputs": outputs_volume,
        "/vol/hf-cache": hf_cache_volume,
        "/vol/repo": repo_volume,
    },
)
def run_generation(
    model_spec: str,
    max_modifiers: int = 10,
    num_samples: int = 8,
    batch_size: int = 4,
    max_tokens: int = 700,
    temperature: float = 0.9,
    top_p: float = 0.95,
    output_subdir: str = "generations",
) -> str:
    repo_root = sync_private_repo_if_configured()
    output_dir = OUTPUTS_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cmd = [
        "python",
        "generate_open_source.py",
        "--model-spec",
        model_spec,
        "--max-modifiers",
        str(max_modifiers),
        "--num-samples",
        str(num_samples),
        "--batch-size",
        str(batch_size),
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--output-dir",
        str(output_dir),
    ]

    print(f"[modal-generate] repo_root={repo_root}")
    print(f"[modal-generate] output_dir={output_dir}")
    print(f"[modal-generate] model_spec={model_spec}")
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    outputs_volume.commit()
    hf_cache_volume.commit()
    repo_volume.commit()
    return str(output_dir)


@app.local_entrypoint()
def main(
    model_spec: str = "qwen3-8b-non-reasoning|Qwen/Qwen3-8B|non_reasoning",
    max_modifiers: int = 10,
    num_samples: int = 8,
    batch_size: int = 4,
    max_tokens: int = 700,
    temperature: float = 0.9,
    top_p: float = 0.95,
    output_subdir: str = "generations",
) -> None:
    out_dir = run_generation.remote(
        model_spec=model_spec,
        max_modifiers=max_modifiers,
        num_samples=num_samples,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        output_subdir=output_subdir,
    )
    print(f"[modal-generate] finished output_dir={out_dir}")
