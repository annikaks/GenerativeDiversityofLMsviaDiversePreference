#!/usr/bin/env python3
"""Run DivPO pair annotation with LLM-as-judge on Modal."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "project-lmdiversity-divpo-judge"
PROJECT_ROOT = Path("/root/project")
OUTPUTS_MOUNT = Path("/vol/outputs")
REPO_SYNC_DIR = Path("/vol/repo")

app = modal.App(APP_NAME)
outputs_volume = modal.Volume.from_name("lmdiversity-outputs", create_if_missing=True)
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
    timeout=60 * 60 * 12,
    volumes={
        str(OUTPUTS_MOUNT): outputs_volume,
        "/vol/repo": repo_volume,
    },
)
def annotate_pairs(
    input_path: str,
    annotated_output_path: str,
    summary_out_path: str = "",
    judge_provider: str = "anthropic",
    judge_model: str = "claude-opus-4-6",
    judge_temperature: float = 0.0,
    rpm: float = 5.0,
    timeout: int = 120,
    retries: int = 4,
) -> str:
    repo_root = sync_private_repo_if_configured()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    def rewrite_output_path(path_str: str) -> str:
        path = Path(path_str)
        try:
            rel = path.resolve().relative_to(Path.cwd().resolve() / "outputs")
            return str(OUTPUTS_MOUNT / rel)
        except Exception:
            pass
        if path_str.startswith("/root/project/outputs/"):
            return path_str.replace("/root/project/outputs", str(OUTPUTS_MOUNT), 1)
        return path_str

    input_path = rewrite_output_path(input_path)
    annotated_output_path = rewrite_output_path(annotated_output_path)
    if summary_out_path:
        summary_out_path = rewrite_output_path(summary_out_path)

    print(f"[modal-judge-pairs] repo_root={repo_root}")
    print(f"[modal-judge-pairs] input={input_path}")
    print(f"[modal-judge-pairs] annotated_output={annotated_output_path}")
    print(f"[modal-judge-pairs] judge_provider={judge_provider} judge_model={judge_model} rpm={rpm}")

    cmd = [
        "python",
        "-u",
        "judge_divpo_pairs.py",
        "--input",
        input_path,
        "--annotated-output",
        annotated_output_path,
        "--judge-provider",
        judge_provider,
        "--judge-model",
        judge_model,
        "--judge-temperature",
        str(judge_temperature),
        "--rpm",
        str(rpm),
        "--timeout",
        str(timeout),
        "--retries",
        str(retries),
    ]
    if summary_out_path:
        cmd.extend(["--summary-out", summary_out_path])

    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    outputs_volume.commit()
    repo_volume.commit()
    return annotated_output_path


@app.local_entrypoint()
def main(
    input_path: str,
    annotated_output_path: str,
    summary_out_path: str = "",
    judge_provider: str = "anthropic",
    judge_model: str = "claude-opus-4-6",
    judge_temperature: float = 0.0,
    rpm: float = 5.0,
    timeout: int = 120,
    retries: int = 4,
) -> None:
    result = annotate_pairs.remote(
        input_path=input_path,
        annotated_output_path=annotated_output_path,
        summary_out_path=summary_out_path,
        judge_provider=judge_provider,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
        rpm=rpm,
        timeout=timeout,
        retries=retries,
    )
    print(f"[modal-judge-pairs] finished annotated_output={result}")
