---
language: en
license: apache-2.0
tags:
  - text-generation
  - onnx
  - pytorch
  - causal-lm
  - nexus-v1
pipeline_tag: text-generation
library_name: transformers
---

# nexus-v1

**nexus-v1** is a lightweight, from-scratch decoder-only causal Transformer language model, built with PyTorch and fully compatible with the Hugging Face `transformers` ecosystem. It ships with an end-to-end MLOps pipeline covering data preparation, training, evaluation, ONNX export with INT8 quantization, and a Gradio playground for interactive testing.

## Model Details

- **Architecture:** Custom mini causal (GPT-style) Transformer, pre-norm decoder blocks with multi-head self-attention and tied input/output embeddings.
- **Parameters:** ~configurable via `config.yaml` (default: 6 layers, 256 hidden size, 8 attention heads — a few million parameters).
- **Tokenizer:** Byte-Level BPE, trained from scratch on the project's dataset.
- **Frameworks:** PyTorch, Hugging Face `transformers`, ONNX Runtime, Gradio.
- **Deployment formats:** PyTorch (`.safetensors`), ONNX (FP32), ONNX (INT8 quantized).

## Intended Use

This model is designed as a **reference implementation / template** for building and deploying small causal language models end-to-end — from raw text to a live Hugging Face Space. The included sample dataset (`data/raw/sample.json`) contains Luau (Roblox scripting language) prompt/completion pairs, making the default trained model suited for small-scale Luau code-completion experiments. Swap in your own dataset to repurpose it for any other domain.

## How to Use

### Load with `transformers` (PyTorch)

```python
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

model = AutoModelForCausalLM.from_pretrained("your-username/nexus-v1", trust_remote_code=True)
tokenizer = PreTrainedTokenizerFast.from_pretrained("your-username/nexus-v1")

inputs = tokenizer("Write a Luau function that adds two numbers.", return_tensors="pt")
outputs = model.generate_greedy_or_sampled(inputs["input_ids"], max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Run with ONNX Runtime (CPU-friendly)

```python
import onnxruntime as ort
import numpy as np
from transformers import PreTrainedTokenizerFast

session = ort.InferenceSession("nexus-v1.int8.onnx", providers=["CPUExecutionProvider"])
tokenizer = PreTrainedTokenizerFast.from_pretrained(".")

input_ids = np.array([tokenizer("Hello,")["input_ids"]], dtype=np.int64)
attention_mask = np.ones_like(input_ids)

logits = session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]
```

## Project Structure

```
nexus-v1/
├── .env.local                 # Template for HF_TOKEN and other secrets (copy to .env)
├── .gitignore                # Excludes cache, weights, and .env from version control
├── config.yaml                # Central hyperparameters, paths, and HF metadata
├── requirements.txt           # ML, HF, ONNX, and Gradio dependencies
├── README.md                  # This file (also serves as the HF model card)
├── data/
│   ├── raw/sample.json        # Sample dataset (Luau prompt/completion pairs)
│   └── processed/             # Cached tokenized dataset
├── models/
│   ├── checkpoints/            # PyTorch checkpoints saved during training
│   └── export/                 # Final artifacts (ONNX, tokenizer, config)
├── src/
│   ├── utils.py                # Logging, seeding, config/.env loading, device detection
│   ├── dataset.py               # Dataset loading, tokenizer training, DataLoader
│   ├── model.py                 # Custom HF-compatible Transformer architecture
│   ├── train.py                 # Full training loop (AMP, checkpointing)
│   ├── evaluate.py               # Loss / perplexity / accuracy evaluation
│   ├── export_onnx.py            # ONNX export + INT8 quantization
│   ├── inference.py               # Local inference testing (PyTorch & ONNX)
│   └── deploy_hf.py                # Automated upload to the Hugging Face Hub
└── app/
    └── app.py                      # Gradio playground for Hugging Face Spaces
```

## Training

```bash
pip install -r requirements.txt
cp .env.local .env   # then fill in your HF_TOKEN, HF_USERNAME, HF_REPO_NAME

python -m src.train              # trains and checkpoints the best model
python -m src.evaluate            # reports loss / perplexity / accuracy
python -m src.export_onnx          # exports + quantizes to ONNX
python -m src.inference --backend onnx --quantized --prompt "Hello"
python -m src.deploy_hf             # uploads models/export/ + README.md to the Hub
```

All hyperparameters (model size, learning rate, batch size, ONNX quantization settings, etc.) are controlled centrally through `config.yaml` — no need to edit source files to tune the run.

## Training Data

The bundled sample dataset (`data/raw/sample.json`) contains a small set of Luau (Roblox scripting language) prompt/completion pairs, intended purely to validate that the full pipeline runs correctly end-to-end. For meaningful results, replace it with a larger, domain-appropriate dataset in the same JSON schema (`{"prompt": ..., "completion": ...}` or `{"text": ...}`).

## Evaluation

Run `python -m src.evaluate` to compute:
- **Loss** — average cross-entropy loss on the held-out validation split.
- **Perplexity** — `exp(loss)`, a standard language-modeling quality metric.
- **Accuracy** — next-token prediction accuracy on non-padded positions.

## Limitations

- The default architecture is intentionally small and is meant as a template/starting point, not a production-grade language model.
- The bundled sample dataset is tiny (8 examples) and exists only to exercise the pipeline; a real training run requires substantially more data.
- As with any language model, outputs may be inaccurate, repetitive, or nonsensical, especially with limited training data.

## License

Released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Citation

If you use this project as a reference implementation, please consider linking back to the Hugging Face Hub repository.
