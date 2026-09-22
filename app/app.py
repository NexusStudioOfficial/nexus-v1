"""
app.py
======
nexus-v1 Interactive Playground -- a Gradio chat interface backed by ONNX
Runtime, designed to run efficiently on the free CPU tier of Hugging Face
Spaces.

The app loads the quantized INT8 ONNX model (falling back to the FP32 model
if the quantized version isn't present) and the matching tokenizer from
models/export/, then exposes a chat UI with temperature, top_p, top_k,
repetition penalty, and max_new_tokens controls.

Run directly (locally or on HF Spaces):
    python app/app.py
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import onnxruntime as ort
from transformers import PreTrainedTokenizerFast

# Make the project root importable when this script is run directly
# (needed both locally and inside an HF Space container).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import get_logger, load_config, resolve_path  # noqa: E402

logger = get_logger(__name__)


# ------------------------------------------------------------------------------
# Custom styling -- kept minimal and dependency-free (no external assets).
# ------------------------------------------------------------------------------
CUSTOM_CSS = """
.nexus-header {
    text-align: center;
    padding: 8px 0 4px 0;
}

.nexus-header h1 {
    margin-bottom: 2px;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.nexus-status {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.9em;
    margin-bottom: 6px;
}

.nexus-status.ok {
    background: var(--color-accent-soft);
}

.nexus-footer {
    text-align: center;
    color: var(--body-text-color-subdued);
    font-size: 0.85em;
    padding-top: 10px;
}
"""

APP_TITLE = "nexus-v1 Playground"
EXAMPLE_PROMPTS = [
    "Write a Luau function that adds two numbers.",
    "Create a Luau function to check if a table is empty.",
    "Write a Luau function that reverses a string.",
]


# ------------------------------------------------------------------------------
# Model loading (once, at startup)
# ------------------------------------------------------------------------------
def load_playground_assets() -> tuple[ort.InferenceSession, PreTrainedTokenizerFast, dict]:
    """
    Load the ONNX Runtime session and tokenizer once at app startup, preferring
    the quantized INT8 model for faster CPU inference and smaller memory usage.

    Returns:
        Tuple of (onnx_session, tokenizer, inference_config). inference_config
        is config.yaml's contents, but with config["model"] replaced by the
        exported model's own config.json -- see the note below.

    Raises:
        FileNotFoundError: If no exported ONNX model can be found.
    """
    config = load_config()
    onnx_cfg = config["onnx"]
    export_dir = resolve_path(onnx_cfg["export_dir"])

    quantized_path = export_dir / onnx_cfg["quantized_filename"]
    fp32_path = export_dir / onnx_cfg["onnx_filename"]

    if quantized_path.exists():
        model_path = quantized_path
        logger.info(f"Using quantized INT8 model: {model_path}")
    elif fp32_path.exists():
        model_path = fp32_path
        logger.info(f"Quantized model not found; falling back to FP32 model: {model_path}")
    else:
        raise FileNotFoundError(
            f"No ONNX model found in '{export_dir}'. "
            f"Run `python -m src.train` then `python -m src.export_onnx` first."
        )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(export_dir))

    # Use the ACTUAL exported model's architecture (models/export/config.json)
    # for anything that must match the ONNX graph in model_path -- not
    # config.yaml's model section. config.yaml describes what src/train.py
    # will build the *next* time it runs; it can drift out of sync with
    # whatever was last trained + exported (e.g. right after editing
    # hidden_size/max_position_embeddings in config.yaml but before
    # re-running train + export_onnx). Reading it back from config.json
    # guarantees this app always reflects the model it actually loaded.
    try:
        with open(export_dir / "config.json", "r", encoding="utf-8") as f:
            exported_model_config = json.load(f)
        config["model"] = exported_model_config
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        logger.warning(
            f"Could not read exported config.json from '{export_dir}' ({exc}); "
            f"falling back to config.yaml's model section, which may not match "
            f"the actually-loaded ONNX model."
        )

    logger.info("nexus-v1 playground assets loaded successfully.")
    return session, tokenizer, config


try:
    ONNX_SESSION, TOKENIZER, CONFIG = load_playground_assets()
    MODEL_MAX_POSITION_EMBEDDINGS: int = CONFIG["model"]["max_position_embeddings"]
    # "name" is a config.yaml-only convenience field, not part of the HF-style
    # config.json schema written by export_onnx.py, so it won't be present
    # after load_playground_assets() swaps config["model"] for config.json's
    # contents. Falls back to the project's fixed name, which is what
    # config.yaml's model.name has always been set to anyway.
    ACTIVE_MODEL_NAME: str = CONFIG.get("model", {}).get("name", "nexus-v1")
    LOAD_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - surface any load failure inside the UI itself
    ONNX_SESSION = None
    TOKENIZER = None
    CONFIG = {}
    MODEL_MAX_POSITION_EMBEDDINGS = 512
    ACTIVE_MODEL_NAME = "nexus-v1"
    LOAD_ERROR = str(exc)
    logger.error(f"Failed to load model assets: {exc}")


# ------------------------------------------------------------------------------
# Generation logic (pure ONNX Runtime, no PyTorch dependency at inference time)
# ------------------------------------------------------------------------------
def generate_response(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """
    Run autoregressive generation against the loaded ONNX session.

    Args:
        prompt: The user's input text.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Softmax temperature for sampling.
        top_p: Nucleus sampling cumulative probability threshold.
        top_k: Top-k sampling cutoff (0 disables top-k filtering).
        repetition_penalty: Penalty factor (>1.0) discouraging token repeats.

    Returns:
        The generated continuation text (decoded, special tokens stripped).
    """
    if ONNX_SESSION is None or TOKENIZER is None:
        return f"Model failed to load: {LOAD_ERROR}"

    if not prompt or not prompt.strip():
        return "Please enter a prompt to generate text."

    input_ids_list: list[int] = TOKENIZER(prompt)["input_ids"]
    eos_token_id = TOKENIZER.eos_token_id
    prompt_len = len(input_ids_list)

    for _ in range(int(max_new_tokens)):
        context = input_ids_list[-MODEL_MAX_POSITION_EMBEDDINGS:]
        input_ids_np = np.array([context], dtype=np.int64)
        attention_mask_np = np.ones_like(input_ids_np, dtype=np.int64)

        ort_inputs = {"input_ids": input_ids_np, "attention_mask": attention_mask_np}
        logits = ONNX_SESSION.run(["logits"], ort_inputs)[0]

        next_token_logits = logits[0, -1, :].astype(np.float64)
        next_token_logits = next_token_logits / max(float(temperature), 1e-5)

        if repetition_penalty != 1.0:
            for token_id in set(input_ids_list):
                next_token_logits[token_id] /= repetition_penalty

        if top_k and top_k > 0:
            k = min(int(top_k), next_token_logits.shape[-1])
            top_k_indices = np.argpartition(next_token_logits, -k)[-k:]
            mask = np.full_like(next_token_logits, -np.inf)
            mask[top_k_indices] = next_token_logits[top_k_indices]
            next_token_logits = mask

        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probs = exp_logits / exp_logits.sum()

        if top_p < 1.0:
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff_idx = int(np.searchsorted(cumulative_probs, top_p) + 1)
            keep_indices = sorted_indices[:cutoff_idx]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[keep_indices] = probs[keep_indices]
            probs = filtered_probs / filtered_probs.sum()

        next_token = int(np.random.choice(len(probs), p=probs))
        input_ids_list.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    generated_ids = input_ids_list[prompt_len:]
    return TOKENIZER.decode(generated_ids, skip_special_tokens=True).strip()


def chat_fn(
    message: str,
    history: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """
    Gradio ChatInterface callback. Builds a lightweight running-context prompt
    from the recent conversation history, then generates a reply.

    Args:
        message: The newest user message.
        history: List of prior turns as {"role": ..., "content": ...} dicts
            (Gradio's "messages" format).
        max_new_tokens: Generation length control from the UI slider.
        temperature: Sampling temperature from the UI slider.
        top_p: Nucleus sampling threshold from the UI slider.
        top_k: Top-k cutoff from the UI slider.
        repetition_penalty: Repetition penalty from the UI slider.

    Returns:
        The model's generated reply as plain text.
    """
    # Keep only the last few turns to stay within the model's small context window.
    recent_turns = history[-4:] if history else []
    context_parts = []
    for turn in recent_turns:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        prefix = "User: " if role == "user" else "Assistant: "
        context_parts.append(f"{prefix}{content}")

    context_parts.append(f"User: {message}")
    context_parts.append("Assistant:")
    full_prompt = "\n".join(context_parts)

    reply = generate_response(
        prompt=full_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    # Trim anything the model hallucinated past a new "User:" turn marker.
    reply = reply.split("User:")[0].strip()
    return reply if reply else "(No output generated. Try adjusting the sampling settings.)"


# ------------------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------------------
def build_interface() -> gr.Blocks:
    """
    Construct the full Gradio Blocks interface: a chat panel plus a sidebar
    of generation-parameter sliders (temperature, top_p, top_k, max tokens,
    repetition penalty).

    Returns:
        A configured gr.Blocks app, ready to `.launch()`.
    """
    default_inf_cfg = CONFIG.get("inference", {}) if CONFIG else {}

    # NOTE: `theme=` and `css=` used to be gr.Blocks() constructor kwargs;
    # Gradio 6.0 moved both to .launch() instead (see the __main__ block
    # at the bottom of this file for the .launch() call).
    with gr.Blocks(title=APP_TITLE, fill_height=True) as demo:
        with gr.Column(elem_classes=["nexus-header"]):
            gr.Markdown(
                f"# {APP_TITLE}\n"
                "A lightweight causal language model running on ONNX Runtime (CPU)."
            )

        if LOAD_ERROR is not None:
            gr.Markdown(
                f'<div class="nexus-status">Model failed to load: {LOAD_ERROR}</div>'
            )
        else:
            gr.Markdown(
                f'<div class="nexus-status ok">'
                f"Model loaded: <strong>{ACTIVE_MODEL_NAME}</strong> "
                f"&middot; Backend: ONNX Runtime (CPU)"
                f"</div>"
            )

        gr.ChatInterface(
            fn=chat_fn,
            # Give the chat log a fixed, comfortable height so only this
            # panel scrolls internally. Without an explicit height, Gradio's
            # default chat box height combined with the header/accordion/
            # examples/footer above and below it makes the *page* taller
            # than the viewport too, so you end up with two nested
            # scrollbars (the chat log's own, plus the browser page's).
            chatbot=gr.Chatbot(height=480),
            additional_inputs=[
                gr.Slider(
                    minimum=1,
                    maximum=256,
                    step=1,
                    value=default_inf_cfg.get("default_max_new_tokens", 64),
                    label="Max new tokens",
                    info="Upper bound on how many tokens the reply can contain.",
                ),
                gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    value=default_inf_cfg.get("default_temperature", 0.8),
                    label="Temperature",
                    info="Higher values increase randomness.",
                ),
                gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    step=0.05,
                    value=default_inf_cfg.get("default_top_p", 0.9),
                    label="Top-p (nucleus sampling)",
                    info="Restricts sampling to the smallest set of tokens exceeding this cumulative probability.",
                ),
                gr.Slider(
                    minimum=0,
                    maximum=200,
                    step=1,
                    value=default_inf_cfg.get("default_top_k", 50),
                    label="Top-k",
                    info="Restricts sampling to the k most likely tokens (0 disables this filter).",
                ),
                gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    step=0.05,
                    value=default_inf_cfg.get("repetition_penalty", 1.2),
                    label="Repetition penalty",
                    info="Values above 1.0 discourage the model from repeating tokens.",
                ),
            ],
            additional_inputs_accordion=gr.Accordion(
                label="Generation settings", open=False
            ),
            # Gradio requires `examples` to be a list of lists when
            # additional_inputs are set: one inner list per example,
            # containing the chat message followed by a value for every
            # additional input, in the same order they were declared above.
            # Passing a flat list of strings (the previous behavior) raises:
            #   ValueError: Examples must be a list of lists when
            #   additional inputs are provided.
            examples=[
                [
                    prompt,
                    default_inf_cfg.get("default_max_new_tokens", 64),
                    default_inf_cfg.get("default_temperature", 0.8),
                    default_inf_cfg.get("default_top_p", 0.9),
                    default_inf_cfg.get("default_top_k", 50),
                    default_inf_cfg.get("repetition_penalty", 1.2),
                ]
                for prompt in EXAMPLE_PROMPTS
            ],
        )

        gr.Markdown(
            f'<div class="nexus-footer">'
            f"Model: {ACTIVE_MODEL_NAME} (custom mini causal Transformer) &middot; "
            f"Backend: ONNX Runtime (INT8 quantized)<br>"
            f"Built with PyTorch, Hugging Face Transformers, and Gradio."
            f"</div>"
        )

    return demo


if __name__ == "__main__":
    interface = build_interface()

    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    requested_port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", 7860)))

    # theme= and css= live here (not in gr.Blocks()) as of Gradio 6.0.
    # server_name/server_port: default to localhost:7860 for local runs;
    # hosting platforms like Render/Railway inject $PORT and expect the app
    # to bind to 0.0.0.0, so those environment variables override the
    # defaults automatically in that environment (see Dockerfile).
    #
    # Railway/Render sometimes start a new container instance before the
    # previous one has fully released its port (a slow shutdown, an
    # overlapping redeploy, or a container that was killed without a clean
    # SIGTERM handler running). When that happens Gradio's own port-picking
    # logic still fails because we're explicitly pinning a single port
    # (server_port=requested_port leaves it no room to pick another one).
    # To make restarts self-healing instead of crash-looping forever, we
    # retry a few times with a short backoff -- giving the old process time
    # to actually die and free the socket -- before giving up loudly.
    import socket
    import time

    def _port_is_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((server_name if server_name != "0.0.0.0" else "0.0.0.0", port))
                return True
            except OSError:
                return False

    max_attempts = 6
    wait_seconds = 5
    for attempt in range(1, max_attempts + 1):
        if _port_is_free(requested_port):
            break
        logger.warning(
            f"Port {requested_port} is still occupied (likely a previous "
            f"instance still shutting down). Retry {attempt}/{max_attempts} "
            f"in {wait_seconds}s..."
        )
        time.sleep(wait_seconds)
    else:
        logger.error(
            f"Port {requested_port} did not free up after {max_attempts} "
            f"retries. If this keeps happening on every deploy, check your "
            f"hosting platform's health-check / restart settings -- the old "
            f"container process may not be receiving or handling SIGTERM."
        )

    interface.launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        server_name=server_name,
        server_port=requested_port,
    )
