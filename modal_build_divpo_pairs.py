#!/usr/bin/env python3
"""Run DivPO pair construction on Modal with persistent volumes."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "project-lmdiversity-divpo-pairs"
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
    .add_local_dir(".", remote_path=str(PROJECT_ROOT))
)


def sync_private_repo_if_configured() -> Path:
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
    timeout=60 * 60 * 12,
    volumes={
        "/vol/outputs": outputs_volume,
        "/vol/hf-cache": hf_cache_volume,
        "/vol/repo": repo_volume,
    },
)
def build_pairs(
    thinking_mode: str = "both",
    samples_per_prompt: int = 8,
    generation_batch_size: int = 4,
    max_new_tokens: int = 700,
    temperature: float = 1.0,
    top_p: float = 0.95,
    embedding_batch_size: int = 64,
    output_subdir: str = "divpo",
) -> str:
    repo_root = sync_private_repo_if_configured()
    output_root = OUTPUTS_DIR / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(HF_CACHE_DIR)
    env["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    cmd = [
        "python",
        "build_divpo_pairs.py",
        "--thinking-mode",
        thinking_mode,
        "--output-root",
        str(output_root),
        "--samples-per-prompt",
        str(samples_per_prompt),
        "--generation-batch-size",
        str(generation_batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--embedding-batch-size",
        str(embedding_batch_size),
    ]

    print(f"[modal-divpo] repo_root={repo_root}")
    print(f"[modal-divpo] output_root={output_root}")
    print(f"[modal-divpo] thinking_mode={thinking_mode}")
    print(
        f"[modal-divpo] samples_per_prompt={samples_per_prompt} "
        f"generation_batch_size={generation_batch_size} max_new_tokens={max_new_tokens}"
    )
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    outputs_volume.commit()
    hf_cache_volume.commit()
    repo_volume.commit()
    return str(output_root)


@app.local_entrypoint()
def main(
    thinking_mode: str = "both",
    samples_per_prompt: int = 8,
    generation_batch_size: int = 4,
    max_new_tokens: int = 700,
    temperature: float = 1.0,
    top_p: float = 0.95,
    embedding_batch_size: int = 64,
    output_subdir: str = "divpo",
) -> None:
    out_dir = build_pairs.remote(
        thinking_mode=thinking_mode,
        samples_per_prompt=samples_per_prompt,
        generation_batch_size=generation_batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        embedding_batch_size=embedding_batch_size,
        output_subdir=output_subdir,
    )
    print(f"[modal-divpo] finished output_root={out_dir}")
