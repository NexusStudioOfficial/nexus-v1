"""
dataset.py
==========
Data loading and preprocessing for nexus-v1.

Responsibilities:
  1. Load the raw JSON dataset (prompt/completion or plain text records).
  2. Train (or load) a byte-pair-encoding tokenizer compatible with Hugging
     Face `transformers` (saved as tokenizer.json, loadable via
     PreTrainedTokenizerFast).
  3. Tokenize and cache the dataset to data/processed/.
  4. Expose a PyTorch Dataset + collate_fn + DataLoader factory for training
     and evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

from src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Raw data loading
# ------------------------------------------------------------------------------
def load_raw_records(raw_path: str | Path) -> list[dict[str, str]]:
    """
    Load the raw JSON dataset file. Expected format is a list of objects, each
    containing either a "text" field, or "prompt" + "completion" fields.

    Args:
        raw_path: Path to the raw JSON file (relative to project root or absolute).

    Returns:
        A list of record dictionaries.

    Raises:
        FileNotFoundError: If the raw data file does not exist.
        ValueError: If the file content is not a JSON list.
    """
    path = resolve_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found at '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of records in '{path}', got {type(data)}.")

    logger.info(f"Loaded {len(data)} raw records from {path}")
    return data


def records_to_texts(
    records: list[dict[str, str]],
    text_field: str = "text",
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    eos_token: str = "<eos>",
) -> list[str]:
    """
    Normalize raw records into a flat list of training strings. Supports two
    schemas: a single `text_field`, or a `prompt_field` + `completion_field`
    pair which gets concatenated with a separating space.

    Args:
        records: Raw record dictionaries.
        text_field: Key name for plain-text records.
        prompt_field: Key name for the prompt half of a pair.
        completion_field: Key name for the completion half of a pair.
        eos_token: Token appended to the end of every training example.

    Returns:
        List of raw training strings, each terminated with `eos_token`.
    """
    texts: list[str] = []
    for record in records:
        if text_field in record and record[text_field]:
            text = record[text_field]
        elif prompt_field in record and completion_field in record:
            text = f"{record[prompt_field]} {record[completion_field]}"
        else:
            logger.warning(f"Skipping malformed record (missing expected fields): {record}")
            continue
        texts.append(f"{text}{eos_token}")

    return texts


# ------------------------------------------------------------------------------
# Tokenizer training / loading
# ------------------------------------------------------------------------------
def train_or_load_tokenizer(
    texts: list[str],
    tokenizer_dir: str | Path,
    vocab_size: int = 8000,
    special_tokens: Optional[dict[str, str]] = None,
    force_retrain: bool = False,
) -> PreTrainedTokenizerFast:
    """
    Train a Byte-Level BPE tokenizer on the provided texts (or load an
    existing one from disk if already trained), then wrap it as a
    Hugging Face `PreTrainedTokenizerFast` for seamless compatibility with
    `transformers` and `AutoModelForCausalLM`.

    Args:
        texts: Corpus of training strings.
        tokenizer_dir: Directory to save/load tokenizer.json from.
        vocab_size: Target vocabulary size.
        special_tokens: Dict with keys pad_token, bos_token, eos_token, unk_token.
        force_retrain: If True, retrain even if a saved tokenizer already exists.

    Returns:
        A PreTrainedTokenizerFast ready for use with the model.
    """
    special_tokens = special_tokens or {
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
    }
    tokenizer_dir = ensure_dir(resolve_path(tokenizer_dir))
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"

    if tokenizer_json_path.exists() and not force_retrain:
        logger.info(f"Loading existing tokenizer from {tokenizer_json_path}")
        fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_json_path),
            pad_token=special_tokens["pad_token"],
            bos_token=special_tokens["bos_token"],
            eos_token=special_tokens["eos_token"],
            unk_token=special_tokens["unk_token"],
        )
        return fast_tokenizer

    logger.info(f"Training new Byte-Level BPE tokenizer (vocab_size={vocab_size})...")

    # Write texts to a temporary corpus file since tokenizers' trainer expects file paths.
    # A sentinel line separator (rather than a raw newline) is used so that
    # meaningful newlines *inside* each example (e.g. code indentation) are
    # preserved and can be learned as part of the BPE vocabulary.
    corpus_path = tokenizer_dir / "_corpus_tmp.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n<|record-sep|>\n")

    raw_tokenizer = ByteLevelBPETokenizer()
    raw_tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=list(special_tokens.values()),
    )

    raw_tokenizer.save(str(tokenizer_json_path))
    corpus_path.unlink(missing_ok=True)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        pad_token=special_tokens["pad_token"],
        bos_token=special_tokens["bos_token"],
        eos_token=special_tokens["eos_token"],
        unk_token=special_tokens["unk_token"],
    )

    logger.info(f"Tokenizer trained and saved to {tokenizer_json_path} (vocab_size={fast_tokenizer.vocab_size})")
    return fast_tokenizer


# ------------------------------------------------------------------------------
# PyTorch Dataset
# ------------------------------------------------------------------------------
class NexusTextDataset(Dataset):
    """
    A PyTorch Dataset that tokenizes a list of raw strings into fixed-length
    input_id / attention_mask tensors suitable for causal language modeling.
    Labels are a copy of input_ids (standard next-token-prediction setup),
    with padded positions masked to -100 so they are ignored by the loss.
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int = 128,
    ) -> None:
        """
        Args:
            texts: List of raw training strings.
            tokenizer: A fitted PreTrainedTokenizerFast.
            max_seq_length: Sequence length to pad/truncate every example to.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.examples: list[dict[str, torch.Tensor]] = self._tokenize_all(texts)

    def _tokenize_all(self, texts: list[str]) -> list[dict[str, torch.Tensor]]:
        encodings = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        examples: list[dict[str, torch.Tensor]] = []
        for i in range(len(texts)):
            input_ids = encodings["input_ids"][i]
            attention_mask = encodings["attention_mask"][i]

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # ignore padded tokens in the loss

            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.examples[idx]


# ------------------------------------------------------------------------------
# Collation & DataLoader factory
# ------------------------------------------------------------------------------
def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Stack a list of tokenized examples into a single padded batch. Since
    NexusTextDataset already pads every example to max_seq_length, this is a
    straightforward stack operation.

    Args:
        batch: List of per-example tensor dictionaries.

    Returns:
        A dictionary of batched tensors: input_ids, attention_mask, labels.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, PreTrainedTokenizerFast]:
    """
    End-to-end factory: loads raw data, trains/loads the tokenizer, splits
    into train/validation sets, and returns ready-to-use DataLoaders.

    Args:
        config: The full parsed config.yaml dictionary.

    Returns:
        Tuple of (train_loader, val_loader, tokenizer).
    """
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    tokenizer_cfg = config["tokenizer"]
    training_cfg = config["training"]

    records = load_raw_records(dataset_cfg["raw_path"])
    texts = records_to_texts(
        records,
        text_field=dataset_cfg.get("text_field", "text"),
        prompt_field=dataset_cfg.get("prompt_field", "prompt"),
        completion_field=dataset_cfg.get("completion_field", "completion"),
        eos_token=tokenizer_cfg["special_tokens"]["eos_token"],
    )

    if len(texts) == 0:
        raise ValueError("No valid training texts found after parsing the raw dataset.")

    tokenizer = train_or_load_tokenizer(
        texts=texts,
        tokenizer_dir=tokenizer_cfg["path"],
        vocab_size=model_cfg["vocab_size"],
        special_tokens=tokenizer_cfg["special_tokens"],
    )

    full_dataset = NexusTextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_length=dataset_cfg["max_seq_length"],
    )

    val_ratio = dataset_cfg.get("val_split_ratio", 0.1)
    val_size = max(1, int(len(full_dataset) * val_ratio)) if len(full_dataset) > 1 else 0
    train_size = len(full_dataset) - val_size

    if val_size == 0:
        logger.warning(
            "Dataset too small for a validation split; using the full dataset "
            "for both training and validation."
        )
        train_subset, val_subset = full_dataset, full_dataset
    else:
        generator = torch.Generator().manual_seed(dataset_cfg.get("seed", 42))
        train_subset, val_subset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        train_subset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=training_cfg["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_cfg.get("num_workers", 0),
        drop_last=False,
    )

    logger.info(
        f"Dataset ready: {len(train_subset)} train examples, "
        f"{len(val_subset)} val examples, vocab_size={tokenizer.vocab_size}"
    )

    return train_loader, val_loader, tokenizer
