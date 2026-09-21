"""
deploy_hf.py
============
Automates publishing the exported nexus-v1 artifacts (models/export/) plus
the model card (README.md) to the Hugging Face Model Hub, using the `HfApi`
client from `huggingface_hub`.

Steps performed:
  1. Validate HF_TOKEN is present and valid.
  2. Create the target repository if it does not already exist.
  3. Upload every file in models/export/ (ONNX weights, tokenizer, config).
  4. Upload README.md as the repo's model card.

Run directly:
    python -m src.deploy_hf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from src.utils import get_env_var, get_hf_token, get_logger, load_config, resolve_path

logger = get_logger(__name__)


def validate_export_directory(export_dir: str | Path) -> Path:
    """
    Ensure the ONNX export directory exists and contains at least one model
    file before attempting an upload.

    Args:
        export_dir: Path to models/export/.

    Returns:
        The resolved, validated export directory path.

    Raises:
        FileNotFoundError: If the directory is missing or empty of model artifacts.
    """
    path = resolve_path(export_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(
            f"Export directory not found at '{path}'. Run `python -m src.export_onnx` first."
        )

    onnx_files = list(path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"No .onnx files found in '{path}'. Run `python -m src.export_onnx` before deploying."
        )

    logger.info(f"Validated export directory {path} ({len(onnx_files)} ONNX file(s) found)")
    return path


def get_or_create_repo(
    api: HfApi,
    repo_id: str,
    repo_type: str = "model",
    private: bool = False,
) -> str:
    """
    Check whether the target repository exists on the Hub; create it if not.

    Args:
        api: An authenticated HfApi client.
        repo_id: Full repo id in "username/repo_name" format.
        repo_type: "model", "dataset", or "space".
        private: Whether to create the repo as private (only applies on creation).

    Returns:
        The repo_id (unchanged), for chaining.
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        logger.info(f"Repository '{repo_id}' already exists on the Hub. Will push updates to it.")
    except RepositoryNotFoundError:
        logger.info(f"Repository '{repo_id}' does not exist yet. Creating it (private={private})...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        logger.info(f"Repository '{repo_id}' created successfully.")

    return repo_id


def upload_export_folder(
    api: HfApi,
    repo_id: str,
    export_dir: str | Path,
    commit_message: str,
    repo_type: str = "model",
) -> None:
    """
    Upload the entire contents of the ONNX export folder to the Hub repo root.

    Args:
        api: An authenticated HfApi client.
        repo_id: Target repo id.
        export_dir: Local directory containing the model artifacts.
        commit_message: Commit message for this upload.
        repo_type: "model", "dataset", or "space".
    """
    export_path = resolve_path(export_dir)

    logger.info(f"Uploading all files from {export_path} to {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(export_path),
        commit_message=commit_message,
        ignore_patterns=["*.pt", "*.pyc", "__pycache__/*", "_corpus_tmp.txt"],
    )
    logger.info("Export folder upload complete.")


def upload_readme(
    api: HfApi,
    repo_id: str,
    readme_path: str | Path,
    commit_message: str,
    repo_type: str = "model",
) -> None:
    """
    Upload README.md as the repository's model card.

    Args:
        api: An authenticated HfApi client.
        repo_id: Target repo id.
        readme_path: Local path to README.md.
        commit_message: Commit message for this upload.
        repo_type: "model", "dataset", or "space".
    """
    path = resolve_path(readme_path)
    if not path.exists():
        logger.warning(f"README.md not found at '{path}'; skipping model card upload.")
        return

    logger.info(f"Uploading model card ({path}) to {repo_id}...")
    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_or_fileobj=str(path),
        path_in_repo="README.md",
        commit_message=commit_message,
    )
    logger.info("Model card upload complete.")


def deploy(config: Optional[dict[str, Any]] = None) -> str:
    """
    Full deployment pipeline: validate token & artifacts, create/verify repo,
    upload export folder and README.md.

    Args:
        config: Optional pre-loaded config dict. If None, loads config.yaml.

    Returns:
        The full repo_id that was deployed to, e.g. "username/nexus-v1".
    """
    config = config or load_config()
    hf_cfg = config["huggingface"]
    onnx_cfg = config["onnx"]

    token = get_hf_token(required=True)
    username = get_env_var(hf_cfg["repo_id_env_username"], required=True)
    repo_name = get_env_var(hf_cfg["repo_name_env"], default=config["model"]["name"])
    private_str = get_env_var(hf_cfg["private_env"], default="false")
    private = str(private_str).strip().lower() in ("true", "1", "yes")

    repo_id = f"{username}/{repo_name}"
    repo_type = hf_cfg.get("repo_type", "model")

    export_dir = validate_export_directory(onnx_cfg["export_dir"])

    api = HfApi(token=token)

    # Validate the token actually works before doing anything else.
    try:
        user_info = api.whoami()
        logger.info(f"Authenticated with Hugging Face Hub as '{user_info.get('name', 'unknown')}'")
    except HfHubHTTPError as e:
        raise ValueError(
            f"HF_TOKEN validation failed: could not authenticate with the Hugging Face Hub. "
            f"Please check that your token is valid and has write access. Original error: {e}"
        ) from e

    get_or_create_repo(api, repo_id=repo_id, repo_type=repo_type, private=private)

    commit_message = hf_cfg.get("commit_message", "Upload nexus-v1 model artifacts")
    upload_export_folder(api, repo_id=repo_id, export_dir=export_dir, commit_message=commit_message, repo_type=repo_type)
    upload_readme(api, repo_id=repo_id, readme_path="README.md", commit_message=commit_message, repo_type=repo_type)

    repo_url = f"https://huggingface.co/{repo_id}"
    logger.info(f"Deployment complete! View your model at: {repo_url}")

    return repo_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy nexus-v1 artifacts to the Hugging Face Hub.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Force-create the repository as private (overrides HF_PRIVATE_REPO env var).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.private:
        import os

        os.environ["HF_PRIVATE_REPO"] = "true"
    deploy()
