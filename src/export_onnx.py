"""
export_onnx.py
==============
Exports a trained nexus-v1 PyTorch checkpoint to ONNX format with dynamic
axes for batch_size and sequence_length, then applies INT8 dynamic
quantization via onnxruntime.quantization to shrink the final artifact for
CPU-only deployment (e.g. free-tier Hugging Face Spaces).

Outputs (written to models/export/):
    - nexus-v1.onnx           (FP32 exported graph)
    - nexus-v1.int8.onnx      (quantized graph)
    - config.json             (HF model config, copied from checkpoint)
    - tokenizer.json / tokenizer files (copied from checkpoint)

Run directly:
    python -m src.export_onnx
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.model import NexusForCausalLM
from src.utils import ensure_dir, get_logger, load_config, resolve_path

logger = get_logger(__name__)


class NexusONNXWrapper(torch.nn.Module):
    """
    Thin wrapper around NexusForCausalLM that returns a raw logits tensor
    instead of a CausalLMOutput dataclass, since torch.onnx.export requires
    the forward pass to return tensors (or a tuple/list of tensors), not
    arbitrary Python objects.
    """

    def __init__(self, model: NexusForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


def export_to_onnx(
    model: NexusForCausalLM,
    export_path: str | Path,
    opset_version: int = 17,
    sample_seq_length: int = 16,
) -> Path:
    """
    Trace and export the model to a standalone ONNX graph with dynamic axes
    on batch_size and sequence_length for both inputs and the output.

    Args:
        model: A trained NexusForCausalLM (already on CPU, in eval mode).
        export_path: Destination file path for the .onnx file.
        opset_version: ONNX opset version to target.
        sample_seq_length: Sequence length used to build the dummy trace inputs.

    Returns:
        Path to the exported .onnx file.
    """
    model.eval()
    wrapped_model = NexusONNXWrapper(model)

    dummy_input_ids = torch.randint(
        low=0, high=model.config.vocab_size, size=(1, sample_seq_length), dtype=torch.long
    )
    dummy_attention_mask = torch.ones((1, sample_seq_length), dtype=torch.long)

    export_path = resolve_path(export_path)
    ensure_dir(export_path.parent)

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    logger.info(f"Exporting model to ONNX (opset={opset_version})...")
    # dynamo=False pins the legacy TorchScript-based exporter, which is
    # stable for this architecture and does not require the optional
    # `onnxscript` dependency that the newer dynamo-based exporter (the
    # default in torch>=2.5) imports at call time.
    torch.onnx.export(
        wrapped_model,
        (dummy_input_ids, dummy_attention_mask),
        str(export_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # Validate the exported graph is well-formed.
    onnx_model = onnx.load(str(export_path))
    onnx.checker.check_model(onnx_model)

    file_size_mb = export_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX export successful: {export_path} ({file_size_mb:.2f} MB)")

    return export_path


def quantize_onnx_model(
    fp32_model_path: str | Path,
    quantized_model_path: str | Path,
    quantization_type: str = "int8",
) -> Path:
    """
    Apply dynamic quantization to an exported ONNX model to reduce its size
    and speed up CPU inference.

    Args:
        fp32_model_path: Path to the source FP32 .onnx file.
        quantized_model_path: Destination path for the quantized .onnx file.
        quantization_type: "int8" (only supported dynamic-quantization mode
            here; fp16 conversion would use onnxconverter_common instead).

    Returns:
        Path to the quantized .onnx file.
    """
    fp32_path = resolve_path(fp32_model_path)
    quantized_path = resolve_path(quantized_model_path)
    ensure_dir(quantized_path.parent)

    if quantization_type.lower() != "int8":
        raise ValueError(
            f"Unsupported quantization_type '{quantization_type}'. "
            f"This export pipeline currently supports 'int8' dynamic quantization."
        )

    logger.info(f"Applying INT8 dynamic quantization to {fp32_path}...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QInt8,
    )

    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    quantized_size_mb = quantized_path.stat().st_size / (1024 * 1024)
    reduction_pct = (1 - quantized_size_mb / fp32_size_mb) * 100 if fp32_size_mb > 0 else 0.0

    logger.info(
        f"Quantization complete: {quantized_path} "
        f"({fp32_size_mb:.2f} MB -> {quantized_size_mb:.2f} MB, {reduction_pct:.1f}% smaller)"
    )

    return quantized_path


def copy_tokenizer_and_config(checkpoint_dir: str | Path, export_dir: str | Path) -> None:
    """
    Copy tokenizer files and the HF config.json from the training checkpoint
    directory into the ONNX export directory, so models/export/ is fully
    self-contained for deployment (matches AutoTokenizer / app.py expectations).

    Args:
        checkpoint_dir: Source checkpoint directory (e.g. models/checkpoints/best_model).
        export_dir: Destination export directory (e.g. models/export).
    """
    checkpoint_path = resolve_path(checkpoint_dir)
    export_path = ensure_dir(resolve_path(export_dir))

    files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for filename in files_to_copy:
        src_file = checkpoint_path / filename
        if src_file.exists():
            shutil.copy2(src_file, export_path / filename)
            logger.info(f"Copied {filename} -> {export_path / filename}")
        else:
            logger.warning(f"Expected file '{filename}' not found in checkpoint directory; skipping.")


def run_export_pipeline(config: Optional[dict[str, Any]] = None, checkpoint_dir: Optional[str] = None) -> dict[str, Path]:
    """
    Full export pipeline: load trained checkpoint -> export ONNX -> quantize
    -> copy tokenizer/config into the export directory.

    Args:
        config: Optional pre-loaded config dict. If None, loads config.yaml.
        checkpoint_dir: Optional override for the source checkpoint directory.

    Returns:
        Dictionary with keys "fp32_path" and "quantized_path" pointing to the
        exported ONNX files.
    """
    config = config or load_config()
    onnx_cfg = config["onnx"]

    checkpoint_dir = checkpoint_dir or (Path(config["training"]["output_dir"]) / "best_model")
    checkpoint_path = resolve_path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint found at '{checkpoint_path}'. Run `python -m src.train` first."
        )

    logger.info(f"Loading trained model from {checkpoint_path} for ONNX export...")
    model = NexusForCausalLM.from_pretrained(str(checkpoint_path))
    model.eval()

    export_dir = onnx_cfg["export_dir"]
    fp32_export_path = Path(export_dir) / onnx_cfg["onnx_filename"]
    quantized_export_path = Path(export_dir) / onnx_cfg["quantized_filename"]

    fp32_path = export_to_onnx(
        model=model,
        export_path=fp32_export_path,
        opset_version=onnx_cfg["opset_version"],
    )

    result_paths = {"fp32_path": fp32_path}

    if onnx_cfg.get("quantize", True):
        quantized_path = quantize_onnx_model(
            fp32_model_path=fp32_path,
            quantized_model_path=quantized_export_path,
            quantization_type=onnx_cfg.get("quantization_type", "int8"),
        )
        result_paths["quantized_path"] = quantized_path

    copy_tokenizer_and_config(checkpoint_path, export_dir)

    logger.info(f"ONNX export pipeline complete. Artifacts available in {resolve_path(export_dir)}")
    return result_paths


if __name__ == "__main__":
    run_export_pipeline()
