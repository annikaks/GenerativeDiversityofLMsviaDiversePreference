#!/usr/bin/env python3
"""Run baseline embedding/diversity and accuracy analysis on Modal.

This wrapper reuses the existing local scripts:
- `pipeline.py analyze-embeddings`
- `pipeline.py analyze-baseline-deviation`
- `run_accuracy_batched.py`

The persistent Modal outputs volume is mounted directly at `/root/project/outputs`,
so the generated JSON artifacts are written in the same paths the local scripts
already expect.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "project-lmdiversity-analysis"
PROJECT_ROOT = Path("/root/project")
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
REPO_SYNC_DIR = Path("/vol/repo")

app = modal.App(APP_NAME)
outputs_volume = modal.Volume.from_name("lmdiversity-outputs", create_if_missing=True)
repo_volume = modal.Volume.from_name("lmdiversity-repo", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path=str(PROJECT_ROOT), ignore=["outputs/", "outputs/**"])
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
    timeout=60 * 60 * 12,
    volumes={
        str(OUTPUTS_PATH): outputs_volume,
        "/vol/repo": repo_volume,
    },
)
def run_analysis(
    task: str = "both",
    embedding_model: str = "intfloat/e5-large",
    judge_provider: str = "anthropic",
    judge_model: str = "claude-opus-4-6",
    batch_size: int = 4,
    rpm: int = 5,
    generation_glob: str = "*.json",
    generation_files_csv: str = "",
) -> str:
    repo_root = sync_private_repo_if_configured()
    generation_files = [p for p in generation_files_csv.split(",") if p]
    print(f"[modal-analysis] repo_root={repo_root}")
    print(f"[modal-analysis] outputs_path={OUTPUTS_PATH}")
    print(f"[modal-analysis] task={task}")
    if generation_files:
        print("[modal-analysis] generation files to analyze:")
        for path in generation_files:
            print(f"[modal-analysis]   {path}")

    if task not in {"diversity", "accuracy", "both"}:
        raise ValueError("task must be one of: diversity, accuracy, both")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if task in {"diversity", "both"}:
        print(f"[modal-analysis] running analyze-embeddings model={embedding_model}")
        embed_cmd = [
            "python",
            "-u",
            "pipeline.py",
            "analyze-embeddings",
            "--embedding-model",
            embedding_model,
        ]
        if generation_files:
            embed_cmd.extend(["--generation-files", *generation_files])
        subprocess.run(
            embed_cmd,
            cwd=str(repo_root),
            env=env,
            check=True,
        )
        print(f"[modal-analysis] running analyze-baseline-deviation model={embedding_model}")
        dev_cmd = [
            "python",
            "-u",
            "pipeline.py",
            "analyze-baseline-deviation",
            "--embedding-model",
            embedding_model,
        ]
        if generation_files:
            dev_cmd.extend(["--generation-files", *generation_files])
        subprocess.run(
            dev_cmd,
            cwd=str(repo_root),
            env=env,
            check=True,
        )
        outputs_volume.commit()

    if task in {"accuracy", "both"}:
        print(
            f"[modal-analysis] running accuracy judge provider={judge_provider} "
            f"model={judge_model} generation_glob={generation_glob}"
        )
        subprocess.run(
            [
                "python",
                "-u",
                "run_accuracy_batched.py",
                "--judge-provider",
                judge_provider,
                "--judge-model",
                judge_model,
                "--batch-size",
                str(batch_size),
                "--rpm",
                str(rpm),
                "--generation-glob",
                generation_glob,
            ],
            cwd=str(repo_root),
            env=env,
            check=True,
        )
        outputs_volume.commit()

    repo_volume.commit()
    return str(OUTPUTS_PATH)


@app.local_entrypoint()
def main(
    task: str = "both",
    embedding_model: str = "intfloat/e5-large",
    judge_provider: str = "anthropic",
    judge_model: str = "claude-opus-4-6",
    batch_size: int = 4,
    rpm: int = 5,
    generation_glob: str = "*.json",
    generation_files_csv: str = "",
) -> None:
    outputs_path = run_analysis.remote(
        task=task,
        embedding_model=embedding_model,
        judge_provider=judge_provider,
        judge_model=judge_model,
        batch_size=batch_size,
        rpm=rpm,
        generation_glob=generation_glob,
        generation_files_csv=generation_files_csv,
    )
    print(f"[modal-analysis] finished outputs_path={outputs_path}")
