"""
evaluate.py
===========
Standalone evaluation script for nexus-v1. Loads a trained checkpoint from
models/checkpoints/best_model and computes:
  - Average cross-entropy loss
  - Perplexity (exp(loss))
  - Next-token prediction accuracy

Run directly:
    python -m src.evaluate
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from src.dataset import build_dataloaders
from src.model import NexusForCausalLM
from src.utils import get_device, get_logger, load_config, resolve_path

logger = get_logger(__name__)


@torch.no_grad()
def compute_metrics(
    model: NexusForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Compute loss, perplexity, and token-level accuracy over a full dataloader.

    Args:
        model: A trained NexusForCausalLM in eval mode.
        dataloader: DataLoader yielding input_ids/attention_mask/labels batches.
        device: Device to run evaluation on.

    Returns:
        Dictionary with keys: "loss", "perplexity", "accuracy".
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

        # Next-token accuracy, matching the shift used inside the model's loss computation.
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        predictions = shift_logits.argmax(dim=-1)
        valid_mask = shift_labels != -100

        total_correct += (predictions == shift_labels).masked_select(valid_mask).sum().item()
        total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / max(1, num_batches)
    perplexity = math.exp(min(avg_loss, 20))  # clamp for numerical stability
    accuracy = total_correct / max(1, total_tokens)

    return {"loss": avg_loss, "perplexity": perplexity, "accuracy": accuracy}


def load_model_for_eval(
    checkpoint_dir: str | Path, device: torch.device
) -> tuple[NexusForCausalLM, PreTrainedTokenizerFast]:
    """
    Load a trained model and its tokenizer from a checkpoint directory.

    Args:
        checkpoint_dir: Path to the saved best_model directory.
        device: Device to move the model to.

    Returns:
        Tuple of (model, tokenizer).
    """
    checkpoint_path = resolve_path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint_path}'. Run `python -m src.train` first."
        )

    model = NexusForCausalLM.from_pretrained(str(checkpoint_path)).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_path))
    model.eval()

    logger.info(f"Loaded model and tokenizer from {checkpoint_path}")
    return model, tokenizer


def evaluate(config: Optional[dict[str, Any]] = None, checkpoint_dir: Optional[str] = None) -> dict[str, float]:
    """
    Full evaluation entrypoint: loads config, model, and validation data, then
    computes and logs metrics.

    Args:
        config: Optional pre-loaded config dict. If None, loads config.yaml.
        checkpoint_dir: Optional override for the checkpoint directory.

    Returns:
        Dictionary of computed metrics.
    """
    config = config or load_config()
    device = get_device()

    checkpoint_dir = checkpoint_dir or (Path(config["training"]["output_dir"]) / "best_model")
    model, _ = load_model_for_eval(checkpoint_dir, device)

    # Rebuild the validation dataloader using the same config-driven pipeline
    # used during training, ensuring consistent preprocessing.
    _, val_loader, _ = build_dataloaders(config)

    metrics = compute_metrics(model, val_loader, device)

    logger.info(
        f"Evaluation results | loss={metrics['loss']:.4f} | "
        f"perplexity={metrics['perplexity']:.2f} | accuracy={metrics['accuracy']:.2%}"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained nexus-v1 checkpoint.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override path to the checkpoint directory (defaults to config.yaml's training.output_dir/best_model).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(checkpoint_dir=args.checkpoint_dir)
