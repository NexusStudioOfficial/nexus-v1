"""
model.py
========
nexus-v1 model architecture: a lightweight, from-scratch causal (decoder-only)
Transformer language model built on PyTorch, structured to be fully compatible
with the Hugging Face `transformers` ecosystem.

The model subclasses `PreTrainedModel` and ships a matching `PretrainedConfig`
subclass, which means:
  - It can be saved with `model.save_pretrained(...)` -> config.json + model.safetensors
  - It can be loaded back with `AutoModelForCausalLM.from_pretrained(...)`
  - It integrates natively with the Hugging Face Hub (push_to_hub, etc.)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


# ------------------------------------------------------------------------------
# Configuration class
# ------------------------------------------------------------------------------
class NexusConfig(PretrainedConfig):
    """
    Configuration class for the Nexus causal language model. Stores every
    architectural hyperparameter needed to reconstruct the model, and is
    serialized to config.json by `save_pretrained`.
    """

    model_type: str = "nexus-causal-lm"

    def __init__(
        self,
        vocab_size: int = 8000,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# ------------------------------------------------------------------------------
# Building blocks
# ------------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention block."""

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.qkv_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Precomputed causal mask buffer, sized to max_position_embeddings.
        # Registered as non-persistent so it is not saved into the checkpoint.
        mask = torch.tril(
            torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding.

        Returns:
            Attention output of shape (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)  # (B, T, 3*H)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask (no attending to future tokens).
        causal = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~causal, float("-inf"))

        # Apply padding mask if provided.
        if attention_mask is not None:
            pad_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)  # (B,1,1,T)
            attn_scores = attn_scores.masked_fill(~pad_mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        # Guard against rows that are fully masked (all -inf -> NaN after softmax).
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerBlock(nn.Module):
    """A single pre-norm Transformer decoder block: Attn -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, config: NexusConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.ln_1(hidden_states), attention_mask)
        hidden_states = hidden_states + self.ffn(self.ln_2(hidden_states))
        return hidden_states


# ------------------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------------------
class NexusForCausalLM(PreTrainedModel):
    """
    nexus-v1: a lightweight decoder-only causal language model.

    Compatible with `AutoModelForCausalLM.from_pretrained(...)` once
    registered, and with `model.save_pretrained(...)` for producing standard
    Hugging Face artifacts (config.json, model.safetensors).
    """

    config_class = NexusConfig
    base_model_prefix = "nexus"
    supports_gradient_checkpointing = False

    # Declares that lm_head.weight is tied to (shares storage with)
    # token_embedding.weight. Newer versions of `transformers` (5.x+) require
    # this explicit mapping to correctly detect and handle shared tensors
    # during save_pretrained(); without it, save_pretrained() raises a
    # RuntimeError because it can't tell the shared tensor is intentional.
    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}

    def __init__(self, config: NexusConfig) -> None:
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights following the standard HF pattern. post_init()
        # internally calls self.tie_weights(), which uses get_input_embeddings()
        # / get_output_embeddings() (defined below) together with
        # _tied_weights_keys to perform the actual weight tying. Doing it this
        # way (rather than a raw `self.lm_head.weight = self.token_embedding.weight`
        # assignment in __init__) keeps PreTrainedModel's bookkeeping in sync,
        # which is what save_pretrained() relies on to detect shared tensors.
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Standard Transformer weight initialization (called by post_init via apply)."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Module:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.token_embedding = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutput:
        """
        Args:
            input_ids: (batch, seq_len) token ids.
            attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding.
            labels: (batch, seq_len) target token ids, with -100 for ignored positions.

        Returns:
            CausalLMOutput with `loss` (if labels provided) and `logits`.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds the model's "
                f"max_position_embeddings ({self.config.max_position_embeddings})."
            )

        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)

        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)  # (B, T, vocab_size)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            # Shift so that tokens < n predict token n.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_greedy_or_sampled(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Lightweight autoregressive text generation loop (used for local sanity
        checks in inference.py; the ONNX/Gradio path in app.py reimplements
        equivalent logic against the exported ONNX graph).

        Args:
            input_ids: (1, seq_len) prompt token ids.
            max_new_tokens: Number of new tokens to generate.
            temperature: Softmax temperature; higher = more random.
            top_p: Nucleus sampling threshold (1.0 disables).
            top_k: Top-k sampling cutoff (0 disables).
            repetition_penalty: Penalty >1.0 discourages repeated tokens.
            eos_token_id: Stop generation early if this token is produced.

        Returns:
            Tensor of shape (1, seq_len + generated_len) with the full sequence.
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_position_embeddings :]
            outputs = self.forward(input_ids=context)
            next_token_logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    next_token_logits[0, token_id] /= repetition_penalty

            if top_k > 0:
                top_k_vals, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                min_val = top_k_vals[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_val, torch.full_like(next_token_logits, float("-inf")), next_token_logits
                )

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float("-inf"))

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return generated
