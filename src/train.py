"""
train.py
========
Full training pipeline for nexus-v1: builds the dataset/dataloaders, builds
the model from config, trains with mixed precision (torch.amp), evaluates
periodically on the validation split, and checkpoints the best model (by
validation loss) to models/checkpoints/.

Run directly:
    python -m src.train
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from src.dataset import build_dataloaders
from src.model import NexusConfig, NexusForCausalLM
from src.utils import count_parameters, ensure_dir, get_device, get_logger, load_config, resolve_path, set_seed

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------------------
def build_model(config: dict[str, Any], actual_vocab_size: Optional[int] = None) -> NexusForCausalLM:
    """
    Instantiate a fresh NexusForCausalLM from the model section of config.yaml.

    Args:
        config: Full parsed config.yaml dictionary.
        actual_vocab_size: The real vocabulary size produced by the trained
            tokenizer. BPE training rarely hits the requested `vocab_size`
            target exactly (it depends entirely on how much text/variety is
            in the corpus) -- e.g. an 8-sentence toy corpus with a
            vocab_size=8000 target may only yield ~600 real merges. If the
            model's embedding/output layer is built at the *requested* size
            instead of the *actual* size, the extra rows are never touched
            by training and stay as random noise that dilutes the softmax at
            every generation step. When provided, this overrides
            model_cfg["vocab_size"] so the model is always sized to match
            what the tokenizer can really produce.

    Returns:
        A freshly initialized NexusForCausalLM.
    """
    model_cfg = config["model"]
    vocab_size = actual_vocab_size if actual_vocab_size is not None else model_cfg["vocab_size"]
    if actual_vocab_size is not None and actual_vocab_size != model_cfg["vocab_size"]:
        logger.warning(
            f"Tokenizer produced vocab_size={actual_vocab_size}, which differs from "
            f"config.yaml's model.vocab_size={model_cfg['vocab_size']}. Building the "
            f"model with the ACTUAL tokenizer size ({actual_vocab_size}) so every "
            f"embedding row is trainable and no dead/random tokens leak into generation."
        )
    nexus_config = NexusConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        dropout=model_cfg["dropout"],
        attention_dropout=model_cfg["attention_dropout"],
        layer_norm_eps=model_cfg["layer_norm_eps"],
        initializer_range=model_cfg["initializer_range"],
        pad_token_id=model_cfg["pad_token_id"],
        bos_token_id=model_cfg["bos_token_id"],
        eos_token_id=model_cfg["eos_token_id"],
        tie_word_embeddings=model_cfg["tie_word_embeddings"],
    )
    model = NexusForCausalLM(nexus_config)
    logger.info(f"Built NexusForCausalLM with {count_parameters(model):,} trainable parameters")
    return model


# ------------------------------------------------------------------------------
# Learning rate schedule
# ------------------------------------------------------------------------------
def build_lr_scheduler(optimizer: torch.optim.Optimizer, num_training_steps: int, warmup_ratio: float) -> LambdaLR:
    """
    Linear warmup followed by linear decay to zero.

    Args:
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of optimizer steps over the whole run.
        warmup_ratio: Fraction of total steps used for warmup.

    Returns:
        A LambdaLR scheduler.
    """
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------------------
# Evaluation loop (used during training for validation)
# ------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model: NexusForCausalLM, dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute the average loss over an entire dataloader without updating weights.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader yielding batches with input_ids/attention_mask/labels.
        device: Device to run evaluation on.

    Returns:
        Mean loss across all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        num_batches += 1

    model.train()
    return total_loss / max(1, num_batches)


# ------------------------------------------------------------------------------
# Checkpoint saving
# ------------------------------------------------------------------------------
def save_checkpoint(
    model: NexusForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    checkpoint_dir: str | Path,
    val_loss: float,
    epoch: int,
    global_step: int,
) -> Path:
    """
    Save a full training checkpoint (model weights via save_pretrained, plus
    a small metadata file with optimizer-independent training state).

    Args:
        model: The model to save.
        tokenizer: Tokenizer to save alongside the model for convenience.
        checkpoint_dir: Directory to save the checkpoint into.
        val_loss: Current best validation loss, recorded in metadata.
        epoch: Current epoch number.
        global_step: Current global training step.

    Returns:
        Path to the checkpoint directory.
    """
    out_dir = ensure_dir(resolve_path(checkpoint_dir) / "best_model")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))

    metadata_path = out_dir / "training_state.pt"
    torch.save(
        {"val_loss": val_loss, "epoch": epoch, "global_step": global_step},
        metadata_path,
    )

    logger.info(f"Saved new best checkpoint (val_loss={val_loss:.4f}) to {out_dir}")
    return out_dir


# ------------------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------------------
def train(config: Optional[dict[str, Any]] = None) -> Path:
    """
    Execute the full training pipeline end to end.

    Args:
        config: Optional pre-loaded config dict. If None, loads config.yaml.

    Returns:
        Path to the directory containing the best saved checkpoint.
    """
    config = config or load_config()
    training_cfg = config["training"]

    set_seed(training_cfg.get("seed", 42))
    device = get_device()

    train_loader, val_loader, tokenizer = build_dataloaders(config)
    # Always build the model against the tokenizer's real vocab size (see
    # build_model's docstring) rather than blindly trusting config.yaml.
    model = build_model(config, actual_vocab_size=tokenizer.vocab_size).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, training_cfg["gradient_accumulation_steps"]))
    num_training_steps = steps_per_epoch * training_cfg["num_epochs"]
    scheduler = build_lr_scheduler(optimizer, num_training_steps, training_cfg["warmup_ratio"])

    use_amp = bool(training_cfg.get("use_amp", True)) and device.type in ("cuda", "cpu")
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device="cuda", enabled=(use_amp and device.type == "cuda"))

    logger.info(
        f"Starting training | device={device} | amp={use_amp} | "
        f"epochs={training_cfg['num_epochs']} | steps/epoch={steps_per_epoch} | "
        f"total_steps={num_training_steps}"
    )

    best_val_loss = float("inf")
    global_step = 0
    patience_counter = 0
    early_stopping_patience = training_cfg.get("early_stopping_patience", 5)
    checkpoint_dir = training_cfg["output_dir"]

    model.train()
    start_time = time.time()

    for epoch in range(1, training_cfg["num_epochs"] + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{training_cfg['num_epochs']}")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / training_cfg["gradient_accumulation_steps"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * training_cfg["gradient_accumulation_steps"]

            if (step + 1) % training_cfg["gradient_accumulation_steps"] == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg["max_grad_norm"])
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % training_cfg.get("log_every_n_steps", 10) == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

                if global_step % training_cfg.get("eval_every_n_steps", 50) == 0:
                    val_loss = evaluate_loss(model, val_loader, device)
                    perplexity = math.exp(min(val_loss, 20))  # clamp to avoid overflow
                    logger.info(
                        f"[step {global_step}] val_loss={val_loss:.4f} | perplexity={perplexity:.2f}"
                    )

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(model, tokenizer, checkpoint_dir, val_loss, epoch, global_step)
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping triggered after {patience_counter} evaluations "
                            f"without improvement."
                        )
                        _finalize_training(model, tokenizer, checkpoint_dir, best_val_loss, start_time)
                        return resolve_path(checkpoint_dir) / "best_model"

        avg_epoch_loss = epoch_loss / max(1, len(train_loader))
        logger.info(f"Epoch {epoch} complete | avg_train_loss={avg_epoch_loss:.4f}")

    # Final end-of-training evaluation & checkpoint (in case eval_every_n_steps
    # never aligned with the very last step, or as a final safety save).
    final_val_loss = evaluate_loss(model, val_loader, device)
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        save_checkpoint(model, tokenizer, checkpoint_dir, final_val_loss, training_cfg["num_epochs"], global_step)

    _finalize_training(model, tokenizer, checkpoint_dir, best_val_loss, start_time)
    return resolve_path(checkpoint_dir) / "best_model"


def _finalize_training(
    model: NexusForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    checkpoint_dir: str | Path,
    best_val_loss: float,
    start_time: float,
) -> None:
    """Log a final training summary and ensure at least one checkpoint exists on disk."""
    elapsed_minutes = (time.time() - start_time) / 60.0
    best_model_dir = resolve_path(checkpoint_dir) / "best_model"

    if not best_model_dir.exists():
        # Safety net: if no checkpoint was ever saved (e.g. extremely short run),
        # save the current model state so downstream scripts have something to load.
        save_checkpoint(model, tokenizer, checkpoint_dir, best_val_loss, epoch=0, global_step=0)

    logger.info(
        f"Training finished in {elapsed_minutes:.1f} min | "
        f"best_val_loss={best_val_loss:.4f} | checkpoint={best_model_dir}"
    )


if __name__ == "__main__":
    train()
