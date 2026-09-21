"""
utils.py
========
Shared utilities for the nexus-v1 pipeline: structured logging, reproducibility
(seeding), configuration loading (config.yaml), secure environment variable
access (.env via python-dotenv), and hardware/device detection.

This module is imported by every other script in src/ and app/app.py, so it
must have zero project-internal dependencies (only stdlib + third-party libs).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from dotenv import load_dotenv

# NOTE: torch is imported lazily inside the functions below (set_seed,
# get_device, count_parameters) instead of at module level. app/app.py
# imports this module just for logging/config/path helpers and never
# touches torch, so keeping it out of the top-level import lets the
# deployed Gradio app run on requirements-app.txt (no PyTorch) with a much
# smaller install. Training scripts that call these functions still import
# torch normally, just at call time instead of at module-load time.

# ------------------------------------------------------------------------------
# Project root resolution
# ------------------------------------------------------------------------------
# utils.py lives at <root>/src/utils.py, so the project root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger with a consistent format across
    the whole project. Safe to call multiple times for the same name; it will
    not attach duplicate handlers.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Seed all relevant random number generators (Python, NumPy, PyTorch CPU/GPU)
    for reproducible results across runs.

    Args:
        seed: The seed value to use everywhere.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Encourage deterministic algorithms where available. This can slightly
    # reduce throughput but greatly improves run-to-run reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Global random seed set to {seed}")


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
def load_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    """
    Load and parse config.yaml into a nested Python dictionary.

    Args:
        config_path: Optional override path to the config file. Defaults to
            <project_root>/config.yaml.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Make sure config.yaml exists at the project root."
        )

    with open(path, "r", encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {path}")
    return config


# ------------------------------------------------------------------------------
# Environment variable / secrets loading
# ------------------------------------------------------------------------------
def load_env(env_path: Optional[str | Path] = None) -> None:
    """
    Load environment variables from a .env file (if present) into os.environ
    using python-dotenv. Safe to call even if the .env file does not exist
    (e.g. in CI/CD or HF Spaces, where secrets are injected directly as real
    environment variables).

    Args:
        env_path: Optional override path to the .env file. Defaults to
            <project_root>/.env.
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH

    if path.exists():
        load_dotenv(dotenv_path=path)
        logger.info(f"Loaded environment variables from {path}")
    else:
        logger.warning(
            f".env file not found at '{path}'. "
            f"Falling back to system environment variables (this is expected "
            f"in CI/CD or Hugging Face Spaces)."
        )


def get_hf_token(required: bool = True) -> Optional[str]:
    """
    Retrieve the Hugging Face access token from the environment.

    Args:
        required: If True, raises a ValueError when the token is missing.
            If False, returns None instead.

    Returns:
        The HF_TOKEN string, or None if not required and not found.

    Raises:
        ValueError: If required=True and HF_TOKEN is not set or is a placeholder.
    """
    load_env()
    token = os.environ.get("HF_TOKEN")

    is_placeholder = token is not None and token.startswith("hf_xxx")

    if not token or is_placeholder:
        if required:
            raise ValueError(
                "HF_TOKEN is not set (or is still the placeholder value). "
                "Please copy .env.example to .env and set a real Hugging Face "
                "token with write access, obtained from "
                "https://huggingface.co/settings/tokens"
            )
        return None

    return token


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Fetch an arbitrary environment variable, loading .env first if needed.

    Args:
        key: Environment variable name.
        default: Value to return if the variable is not set.
        required: If True, raises ValueError when the variable is missing.

    Returns:
        The variable's string value, or `default`.

    Raises:
        ValueError: If required=True and the variable is not set.
    """
    load_env()
    value = os.environ.get(key, default)

    if required and not value:
        raise ValueError(f"Required environment variable '{key}' is not set.")

    return value


# ------------------------------------------------------------------------------
# Device detection
# ------------------------------------------------------------------------------
def get_device(override: Optional[str] = None) -> torch.device:
    """
    Automatically detect the best available compute device, in priority order:
    CUDA GPU -> Apple Silicon MPS -> CPU. An explicit override always wins.

    Args:
        override: Optional explicit device string (e.g. "cuda", "mps", "cpu").
            If None, checks the DEVICE_OVERRIDE environment variable, then
            auto-detects.

    Returns:
        A torch.device instance ready to use for tensors/models.
    """
    import torch

    if override is None:
        override = get_env_var("DEVICE_OVERRIDE", default=None)

    if override:
        device = torch.device(override)
        logger.info(f"Using explicitly overridden device: {device}")
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple Silicon MPS backend detected.")
    else:
        device = torch.device("cpu")
        logger.info("No GPU detected, falling back to CPU.")

    return device


def get_onnx_providers(override: Optional[str] = None) -> list[str]:
    """
    Pick the best available ONNX Runtime execution provider(s), in priority
    order: CUDA GPU -> CPU. An explicit override always wins.

    Requires the `onnxruntime-gpu` package to actually use CUDA (the plain
    `onnxruntime` package only ships the CPU provider) -- see requirements.txt.
    `onnxruntime` and `onnxruntime-gpu` install the same `onnxruntime` module
    name and conflict with each other, so only one of the two should be
    installed at a time: `pip uninstall onnxruntime && pip install onnxruntime-gpu`
    to enable GPU, or the reverse to go back to CPU-only.

    Args:
        override: Optional explicit provider string ("cuda" or "cpu"). If
            None, checks the DEVICE_OVERRIDE environment variable, then
            auto-detects based on what's actually available.

    Returns:
        A list of ONNX Runtime provider names, most-preferred first, ready
        to pass straight into `onnxruntime.InferenceSession(..., providers=...)`.
    """
    import onnxruntime as ort

    if override is None:
        override = get_env_var("DEVICE_OVERRIDE", default=None)

    available = ort.get_available_providers()

    if override:
        override = override.lower()
        if override in ("cuda", "gpu"):
            if "CUDAExecutionProvider" not in available:
                logger.warning(
                    "DEVICE_OVERRIDE requested CUDA but 'CUDAExecutionProvider' is not "
                    "available in this onnxruntime install (need onnxruntime-gpu + a "
                    "working CUDA/cuDNN setup). Falling back to CPU."
                )
                return ["CPUExecutionProvider"]
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        logger.info("CUDA GPU detected for ONNX Runtime (CUDAExecutionProvider).")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    logger.info("No GPU execution provider available for ONNX Runtime, using CPU.")
    return ["CPUExecutionProvider"]


# ------------------------------------------------------------------------------
# Filesystem helpers
# ------------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating parent directories as needed.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(relative_path: str | Path) -> Path:
    """
    Resolve a path relative to the project root, so scripts work correctly
    regardless of the current working directory they are invoked from.

    Args:
        relative_path: A path relative to the project root (e.g. "data/raw/sample.json").

    Returns:
        An absolute Path.
    """
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: The model to inspect.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total parameter count as an integer.
    """
    # No torch import needed here: the method calls below run on the
    # already-constructed `model` object passed in by the caller.
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
