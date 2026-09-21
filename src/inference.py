"""
inference.py
============
Local inference sanity-check script for nexus-v1. Supports running generation
through either:
  1. The raw PyTorch checkpoint (models/checkpoints/best_model), or
  2. The exported ONNX Runtime graph (models/export/*.onnx)

This lets you verify that the ONNX export produces output consistent with
the original PyTorch model before deploying.

Run directly:
    python -m src.inference --backend pytorch --prompt "Hello world"
    python -m src.inference --backend onnx --prompt "Hello world" --quantized
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
import torch
from transformers import PreTrainedTokenizerFast

from src.model import NexusForCausalLM
from src.utils import get_device, get_logger, get_onnx_providers, load_config, resolve_path

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# PyTorch backend
# ------------------------------------------------------------------------------
def run_pytorch_inference(
    prompt: str,
    checkpoint_dir: str | Path,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    device_override: Optional[str] = None,
) -> str:
    """
    Generate text using the raw PyTorch checkpoint.

    Args:
        prompt: The input text prompt.
        checkpoint_dir: Path to the trained checkpoint directory.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalty for repeated tokens.
        device_override: Force a specific device ("cuda", "mps", "cpu"). If
            None, auto-detects the best available device (CUDA -> MPS -> CPU).

    Returns:
        The generated text (prompt + continuation).
    """
    device = get_device(override=device_override)
    checkpoint_path = resolve_path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint_path}'. Run `python -m src.train` first."
        )

    model = NexusForCausalLM.from_pretrained(str(checkpoint_path)).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(checkpoint_path))
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    output_ids = model.generate_greedy_or_sampled(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


# ------------------------------------------------------------------------------
# ONNX Runtime backend
# ------------------------------------------------------------------------------
def load_onnx_session(onnx_path: str | Path, device_override: Optional[str] = None) -> ort.InferenceSession:
    """
    Load an ONNX model into an ONNX Runtime inference session, preferring a
    CUDA GPU execution provider when available and falling back to CPU
    (CPU is always fine and is what's used on free-tier Hugging Face Spaces).

    Args:
        onnx_path: Path to the .onnx file.
        device_override: Force a specific provider ("cuda" or "cpu"). If
            None, auto-detects (CUDA -> CPU). Requires the `onnxruntime-gpu`
            package to actually use CUDA -- see requirements.txt.

    Returns:
        A ready-to-use ort.InferenceSession.
    """
    path = resolve_path(onnx_path)
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found at '{path}'. Run `python -m src.export_onnx` first.")

    providers = get_onnx_providers(override=device_override)
    session = ort.InferenceSession(str(path), providers=providers)
    logger.info(f"Loaded ONNX Runtime session from {path} (providers={session.get_providers()})")
    return session


def onnx_generate(
    session: ort.InferenceSession,
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    max_position_embeddings: int = 512,
) -> str:
    """
    Autoregressive text generation driven entirely by an ONNX Runtime session
    (no PyTorch model object involved), matching the sampling behavior used
    in the PyTorch generation loop.

    Args:
        session: An ONNX Runtime InferenceSession for the exported model.
        tokenizer: The matching PreTrainedTokenizerFast.
        prompt: Input text prompt.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalty for repeated tokens.
        max_position_embeddings: Max context length supported by the model.

    Returns:
        The generated text (prompt + continuation).
    """
    input_ids_list: list[int] = tokenizer(prompt)["input_ids"]
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        context = input_ids_list[-max_position_embeddings:]
        input_ids_np = np.array([context], dtype=np.int64)
        attention_mask_np = np.ones_like(input_ids_np, dtype=np.int64)

        ort_inputs = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
        logits = session.run(["logits"], ort_inputs)[0]  # (1, seq_len, vocab_size)

        next_token_logits = logits[0, -1, :].astype(np.float64)
        next_token_logits = next_token_logits / max(temperature, 1e-5)

        if repetition_penalty != 1.0:
            for token_id in set(input_ids_list):
                next_token_logits[token_id] /= repetition_penalty

        if top_k > 0:
            top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
            mask = np.full_like(next_token_logits, -np.inf)
            mask[top_k_indices] = next_token_logits[top_k_indices]
            next_token_logits = mask

        # Softmax
        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probs = exp_logits / exp_logits.sum()

        if top_p < 1.0:
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff_idx = np.searchsorted(cumulative_probs, top_p) + 1
            keep_indices = sorted_indices[:cutoff_idx]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[keep_indices] = probs[keep_indices]
            probs = filtered_probs / filtered_probs.sum()

        next_token = int(np.random.choice(len(probs), p=probs))
        input_ids_list.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokenizer.decode(input_ids_list, skip_special_tokens=True)


def run_onnx_inference(
    prompt: str,
    export_dir: str | Path,
    use_quantized: bool = True,
    onnx_filename: str = "nexus-v1.onnx",
    quantized_filename: str = "nexus-v1.int8.onnx",
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    device_override: Optional[str] = None,
) -> str:
    """
    Generate text using the exported ONNX model (FP32 or quantized INT8).

    Args:
        prompt: Input text prompt.
        export_dir: Directory containing the exported .onnx files and tokenizer.
        use_quantized: If True, use the quantized INT8 model; else FP32.
        onnx_filename: Filename of the FP32 ONNX model.
        quantized_filename: Filename of the quantized ONNX model.
        max_new_tokens: Number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        repetition_penalty: Penalty for repeated tokens.
        device_override: Force a specific provider ("cuda" or "cpu") for
            ONNX Runtime. If None, auto-detects.

    Returns:
        The generated text (prompt + continuation).
    """
    export_path = resolve_path(export_dir)
    filename = quantized_filename if use_quantized else onnx_filename
    onnx_path = export_path / filename

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(export_path))
    session = load_onnx_session(onnx_path, device_override=device_override)

    # Read the model's actual max context length from its exported config.json
    # (models/export/config.json) instead of relying on onnx_generate's 512
    # default -- that default silently under-uses the model whenever the
    # architecture's max_position_embeddings differs from 512 (e.g. after
    # increasing it in config.yaml), truncating context earlier than the
    # model actually supports.
    try:
        with open(export_path / "config.json", "r", encoding="utf-8") as f:
            exported_config = json.load(f)
        max_position_embeddings = exported_config["max_position_embeddings"]
    except Exception:
        logger.warning(
            "Could not read max_position_embeddings from the exported config.json; "
            "falling back to onnx_generate's default of 512."
        )
        max_position_embeddings = 512

    return onnx_generate(
        session=session,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_position_embeddings=max_position_embeddings,
    )


# ------------------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference for nexus-v1.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input text prompt.")
    parser.add_argument(
        "--backend", type=str, choices=["pytorch", "onnx"], default="pytorch", help="Inference backend to use."
    )
    parser.add_argument("--quantized", action="store_true", help="Use the quantized INT8 ONNX model (onnx backend only).")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Force a specific device/provider. Default: auto-detect (CUDA/MPS GPU if available, else CPU).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max new tokens.")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Override nucleus sampling top_p.")
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k sampling.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config()
    inference_cfg = config["inference"]

    max_new_tokens = args.max_new_tokens or inference_cfg["default_max_new_tokens"]
    temperature = args.temperature if args.temperature is not None else inference_cfg["default_temperature"]
    top_p = args.top_p if args.top_p is not None else inference_cfg["default_top_p"]
    top_k = args.top_k if args.top_k is not None else inference_cfg["default_top_k"]
    repetition_penalty = inference_cfg["repetition_penalty"]

    if args.backend == "pytorch":
        checkpoint_dir = Path(config["training"]["output_dir"]) / "best_model"
        result = run_pytorch_inference(
            prompt=args.prompt,
            checkpoint_dir=checkpoint_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device_override=args.device,
        )
    else:
        result = run_onnx_inference(
            prompt=args.prompt,
            export_dir=config["onnx"]["export_dir"],
            use_quantized=args.quantized,
            onnx_filename=config["onnx"]["onnx_filename"],
            quantized_filename=config["onnx"]["quantized_filename"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            device_override=args.device,
        )

    print("\n" + "=" * 80)
    print(f"PROMPT: {args.prompt}")
    print("-" * 80)
    print(f"OUTPUT: {result}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
