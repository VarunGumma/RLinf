# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0-Fast: autoregressive embodied policy using PaliGemma + FAST tokenizer.

Unlike Pi0/Pi0.5, this model has no separate "action expert" stream.  Continuous
actions are encoded as discrete tokens via the FAST tokenizer and predicted
autoregressively with cross-entropy loss (not flow matching).
"""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor

# PaliGemma EOS token id
_PALIGEMMA_EOS_TOKEN_ID = 1
# PaliGemma vocab size
_PALIGEMMA_VOCAB_SIZE = 257152

# Parameter name substrings that must stay in float32 even when the rest of
# the model is cast to bfloat16 (mirrors openpi's precision handling).
_PLACEHOLDER_PROMPT = "xxxx"

_PARAMS_TO_KEEP_FLOAT32 = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


@dataclasses.dataclass
class OpenPiFastConfig:
    """Configuration for the Pi0-Fast model in RLinf.

    Attributes:
        config_name: openpi config name used for data loading.
        num_images_in_input: Number of images to consume from the observation.
        action_chunk: Action horizon (number of action steps to predict).
        action_env_dim: Environment action dimensionality.
        max_token_len: Maximum prefix token length (padded to this size).
        max_new_tokens: Maximum number of tokens to generate at inference.
        paligemma_variant: PaliGemma size variant, e.g. ``"gemma_2b"``.
        add_value_head: Whether to add an MLP value head for PPO critic.
        train_expert_only: If True, freeze PaliGemma and only train added heads.
        noise_method: Kept for config compatibility with openpi machinery.
        logprob_type: Log-prob type consumed by the PPO actor worker.
        single_action_dim: Must be 1 for chunk-level log-prob compatibility.
    """

    config_name: str = "pi0_fast_libero"
    num_images_in_input: int = 2
    action_chunk: int = 10
    action_env_dim: int = 7
    max_token_len: int = 250
    max_new_tokens: int = 128
    paligemma_variant: str = "gemma_2b"
    add_value_head: bool = False
    train_expert_only: bool = True
    noise_method: str = "none"
    logprob_type: str = "chunk_level"
    single_action_dim: int = 1


class OpenPiFastPytorch(nn.Module):
    """PyTorch implementation of Pi0-Fast.

    Architecture: PaliGemma only (SigLIP vision encoder + Gemma LM).
    Actions are tokenised via the FAST tokeniser and predicted autoregressively
    with a prefix-LM attention pattern.
    """

    def __init__(self, config: OpenPiFastConfig) -> None:
        super().__init__()
        self.config = config
        self.logger = get_logger()

        # ------------------------------------------------------------------
        # Build PaliGemma from openpi's architecture config tables
        # ------------------------------------------------------------------
        import openpi.models.gemma_fast as _gemma_fast

        paligemma_cfg = _gemma_fast.get_config(config.paligemma_variant)

        vlm_hf = CONFIG_MAPPING["paligemma"]()
        vlm_hf._vocab_size = _PALIGEMMA_VOCAB_SIZE  # noqa: SLF001
        vlm_hf.image_token_index = _PALIGEMMA_VOCAB_SIZE
        vlm_hf.text_config.hidden_size = paligemma_cfg["width"]
        vlm_hf.text_config.intermediate_size = paligemma_cfg["mlp_dim"]
        vlm_hf.text_config.num_attention_heads = paligemma_cfg["num_heads"]
        vlm_hf.text_config.head_dim = paligemma_cfg["head_dim"]
        vlm_hf.text_config.num_hidden_layers = paligemma_cfg["depth"]
        vlm_hf.text_config.num_key_value_heads = paligemma_cfg["num_kv_heads"]
        vlm_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_hf.text_config.torch_dtype = "float32"
        vlm_hf.text_config.vocab_size = _PALIGEMMA_VOCAB_SIZE
        vlm_hf.vision_config.intermediate_size = 4304
        vlm_hf.vision_config.projection_dim = 2048
        vlm_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_hf.vision_config.torch_dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_hf)
        # Eager attention is required for custom 2-D attention masks.
        self.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # ------------------------------------------------------------------
        # FAST tokeniser (decoding only; tokenisation happens in the data pipe)
        # ------------------------------------------------------------------
        from openpi.models.tokenizer import FASTTokenizer

        self._fast_tokenizer = FASTTokenizer(config.max_token_len)

        # ------------------------------------------------------------------
        # Optional value head for critic-based RL (e.g. PPO)
        # ------------------------------------------------------------------
        if config.add_value_head:
            self.value_head = ValueHead(
                input_dim=paligemma_cfg["width"],
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        # Cast to bfloat16, keeping selected parameters in float32.
        self._set_precision("bfloat16")

        self.gradient_checkpointing_enabled = False

        for name, module in self.named_modules():
            parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", parts[-1] if parts else name)

    # ------------------------------------------------------------------
    # Precision helpers
    # ------------------------------------------------------------------

    def _set_precision(self, precision: str = "bfloat16") -> None:
        """Cast model to bfloat16, keeping selected norm/embedding params in fp32."""
        if precision != "bfloat16":
            return
        self.to(dtype=torch.bfloat16)
        for name, param in self.named_parameters():
            if any(sel in name for sel in _PARAMS_TO_KEEP_FLOAT32):
                param.data = param.data.to(dtype=torch.float32)

    def to_bfloat16_for_selected_params(self, precision: str = "bfloat16") -> None:
        """Public alias kept for API compatibility with the openpi model pattern."""
        self._set_precision(precision)

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing for the language model and vision tower."""
        self.gradient_checkpointing_enabled = True
        self.paligemma.language_model.gradient_checkpointing = True
        if hasattr(self.paligemma, "vision_tower"):
            self.paligemma.vision_tower.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma.language_model.gradient_checkpointing = False
        if hasattr(self.paligemma, "vision_tower"):
            self.paligemma.vision_tower.gradient_checkpointing = False

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a single image batch with SigLIP.

        Args:
            image: Float tensor of shape ``[B, C, H, W]``.

        Returns:
            Image patch embeddings of shape ``[B, N_img, D]``.
        """
        return self.paligemma.model.get_image_features(image)

    def _embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed integer token IDs and apply the standard LM embedding scaling.

        Args:
            tokens: Long tensor of shape ``[B, T]``.

        Returns:
            Scaled embeddings of shape ``[B, T, D]``.
        """
        emb = self.paligemma.language_model.embed_tokens(tokens)
        return emb * math.sqrt(emb.shape[-1])

    # ------------------------------------------------------------------
    # Prefix embedding
    # ------------------------------------------------------------------

    def _embed_prefix(
        self,
        images: list[torch.Tensor],
        img_masks: list[torch.Tensor],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_ar_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and text tokens into a unified prefix sequence.

        Args:
            images: List of image tensors, each ``[B, C, H, W]``.
            img_masks: Per-batch image validity masks, each ``[B]`` bool.
            text_tokens: Integer token IDs ``[B, T]``.
            text_mask: Boolean validity mask ``[B, T]``.
            text_ar_mask: Autoregressive mask ``[B, T]`` long
                (0 = bidirectional prefix, 1 = causal action postfix).

        Returns:
            embs:      Combined embeddings ``[B, S, D]`` where S = N_img_total + T.
            pad_masks: Boolean validity mask ``[B, S]``.
            att_masks: Long autoregressive mask ``[B, S]`` (0/1).
        """
        bsz = text_tokens.shape[0]
        device = text_tokens.device

        embs_list: list[torch.Tensor] = []
        pad_list: list[torch.Tensor] = []
        total_img_tokens = 0

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._embed_image(img)  # [B, N_img, D]
            n_img = img_emb.shape[1]
            embs_list.append(img_emb)
            pad_list.append(img_mask[:, None].expand(bsz, n_img).bool())
            total_img_tokens += n_img

        lang_emb = self._embed_language_tokens(text_tokens)  # [B, T, D]
        embs_list.append(lang_emb)
        pad_list.append(text_mask.bool())

        embs = torch.cat(embs_list, dim=1)
        pad_masks = torch.cat(pad_list, dim=1)

        # Image tokens are always bidirectional (ar_mask = 0).
        img_ar = torch.zeros(bsz, total_img_tokens, dtype=torch.long, device=device)
        att_masks = torch.cat([img_ar, text_ar_mask.long()], dim=1)

        return embs, pad_masks, att_masks

    # ------------------------------------------------------------------
    # Attention mask construction
    # ------------------------------------------------------------------

    def _prepare_attn_mask_4d(
        self,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Build a 4-D additive attention mask ``[B, 1, S, S]``.

        Allowed positions map to 0.0; masked positions map to -inf.

        Args:
            pad_masks: Boolean validity mask ``[B, S]``.
            att_masks: Long autoregressive mask ``[B, S]``.

        Returns:
            Float attention bias tensor ``[B, 1, S, S]``.
        """
        att_2d = make_att_2d_masks(pad_masks, att_masks)  # [B, S, S] bool
        att_4d = att_2d[:, None, :, :]
        return torch.where(att_4d, 0.0, torch.finfo(torch.float32).min).float()

    # ------------------------------------------------------------------
    # SFT training forward (teacher forcing)
    # ------------------------------------------------------------------

    def forward(
        self,
        observation: "_model.Observation",
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """SFT forward pass.

        Args:
            observation: Preprocessed observation containing tokenised prompt
                fields produced by ``TokenizeFASTInputs``.
            actions: Unused; kept for API compatibility with the BasePolicy
                SFT interface.

        Returns:
            Scalar mean cross-entropy loss over action token positions.
        """
        return self._compute_sft_loss(observation)

    def _compute_sft_loss(self, observation: "_model.Observation") -> torch.Tensor:
        """Compute teacher-forcing CE loss over action token positions.

        The full token sequence (prefix + "Action: " + FAST action tokens +
        "|" + EOS) is expected in ``observation.tokenized_prompt``, produced by
        ``TokenizeFASTInputs`` in the data pipeline.

        Args:
            observation: Preprocessed observation with tokenised prompt fields.

        Returns:
            Scalar cross-entropy loss.
        """
        images = list(observation.images.values())[: self.config.num_images_in_input]
        img_masks = list(observation.image_masks.values())[: self.config.num_images_in_input]

        text_tokens = observation.tokenized_prompt  # [B, T]
        text_mask = observation.tokenized_prompt_mask  # [B, T]

        token_ar_mask = getattr(observation, "token_ar_mask", None)
        if token_ar_mask is None:
            token_ar_mask = torch.zeros_like(text_tokens, dtype=torch.long)

        token_loss_mask = getattr(observation, "token_loss_mask", None)
        if token_loss_mask is None:
            token_loss_mask = token_ar_mask.bool()

        embs, pad_masks, att_masks = self._embed_prefix(
            images, img_masks, text_tokens, text_mask, token_ar_mask.long()
        )
        S = embs.shape[1]
        T = text_tokens.shape[1]
        num_img_tokens = S - T

        # Teacher-forcing: input is embs[:, :-1], target is the next token.
        att_4d = self._prepare_attn_mask_4d(pad_masks[:, :-1], att_masks[:, :-1])
        position_ids = (torch.cumsum(pad_masks[:, :-1].long(), dim=1) - 1).clamp(min=0)

        if self.gradient_checkpointing_enabled and self.training:

            def _lm_fwd(
                embs: torch.Tensor,
                att_4d: torch.Tensor,
                pos_ids: torch.Tensor,
            ) -> torch.Tensor:
                return self.paligemma.language_model(
                    inputs_embeds=embs[:, :-1],
                    attention_mask=att_4d,
                    position_ids=pos_ids,
                    use_cache=False,
                ).last_hidden_state

            hidden = torch.utils.checkpoint.checkpoint(
                _lm_fwd, embs, att_4d, position_ids, use_reentrant=False
            )
        else:
            hidden = self.paligemma.language_model(
                inputs_embeds=embs[:, :-1],
                attention_mask=att_4d,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state  # [B, S-1, D]

        # hidden[:, num_img_tokens - 1 + k, :] predicts text_tokens[:, k]
        # for k in 0 .. T-1 (the LM sees img and text_tokens[:k] → predicts token k).
        text_hidden = hidden[:, num_img_tokens - 1 : num_img_tokens - 1 + T, :]  # [B, T, D]

        logits = self.paligemma.language_model.lm_head(
            text_hidden.to(torch.bfloat16)
        ).float()  # [B, T, V]

        B, V = logits.shape[0], logits.shape[-1]
        targets = text_tokens.long()  # [B, T]

        ce_per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="none",
        ).reshape(B, T)

        loss_mask_f = token_loss_mask.float()
        denom = loss_mask_f.sum(dim=-1).clamp(min=1.0)
        loss = (ce_per_token * loss_mask_f).sum(dim=-1) / denom
        return loss.mean()

    # ------------------------------------------------------------------
    # Inference: autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        observation: "_model.Observation",
        *,
        temperature: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Autoregressively generate action tokens and decode to continuous actions.

        Args:
            observation: Preprocessed observation (prefix context only; no
                action tokens in the sequence yet).
            temperature: Softmax temperature for token sampling.  Use 0 for
                greedy argmax decoding.
            max_new_tokens: Override the config default if provided.

        Returns:
            Dict with keys:

            - ``actions``:          ``[B, action_chunk, action_env_dim]`` float32.
            - ``generated_tokens``: ``[B, max_new_tokens]`` long.
            - ``token_log_probs``:  ``[B, max_new_tokens]`` float32.
            - ``token_mask``:       ``[B, max_new_tokens]`` bool (True = real token).
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        device = next(self.parameters()).device

        images = list(observation.images.values())[: self.config.num_images_in_input]
        img_masks = list(observation.image_masks.values())[: self.config.num_images_in_input]

        text_tokens = observation.tokenized_prompt  # [B, T]
        text_mask = observation.tokenized_prompt_mask  # [B, T]
        token_ar_mask = getattr(observation, "token_ar_mask", None)
        if token_ar_mask is None:
            token_ar_mask = torch.zeros_like(text_tokens, dtype=torch.long)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._embed_prefix(
            images, img_masks, text_tokens, text_mask, token_ar_mask.long()
        )
        bsz = prefix_embs.shape[0]
        prefix_len = prefix_embs.shape[1]
        kv_len = prefix_len + max_new_tokens

        # Build 2-D bidirectional prefix attention mask and extend for the KV
        # cache slots that will be filled by generated tokens.
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_ext = torch.zeros(bsz, prefix_len, kv_len, dtype=torch.bool, device=device)
        prefix_att_ext[:, :, :prefix_len] = prefix_att_2d

        att_4d_prefix = torch.where(
            prefix_att_ext[:, None, :, :], 0.0, torch.finfo(torch.float32).min
        ).float()  # [B, 1, P, kv_len]

        prefix_pos_ids = (torch.cumsum(prefix_pad_masks.long(), dim=1) - 1).clamp(min=0)

        self.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Prefill: forward pass on the full prefix to populate the KV cache.
        lm_out = self.paligemma.language_model(
            inputs_embeds=prefix_embs,
            attention_mask=att_4d_prefix,
            position_ids=prefix_pos_ids,
            past_key_values=None,
            use_cache=True,
        )
        last_hidden = lm_out.last_hidden_state  # [B, P, D]
        kv_cache = lm_out.past_key_values

        # The hidden state at the last valid prefix position predicts the first
        # generated token.
        valid_counts = prefix_pad_masks.long().sum(dim=1)  # [B]
        last_valid_idx = (valid_counts - 1).clamp(min=0)  # [B]
        last_h = last_hidden[torch.arange(bsz, device=device), last_valid_idx]  # [B, D]
        next_logit = self.paligemma.language_model.lm_head(
            last_h[:, None].to(torch.bfloat16)
        ).float()  # [B, 1, V]

        gen_tokens = torch.zeros(bsz, max_new_tokens, dtype=torch.long, device=device)
        gen_log_probs = torch.zeros(bsz, max_new_tokens, dtype=torch.float32, device=device)
        gen_mask = torch.zeros(bsz, max_new_tokens, dtype=torch.bool, device=device)
        all_eos = torch.zeros(bsz, dtype=torch.bool, device=device)

        # Per-sample next absolute position id (just after the last valid prefix token).
        cur_pos_ids = valid_counts.clone()  # [B]

        for step in range(max_new_tokens):
            logit_step = next_logit[:, -1, :]  # [B, V]

            if temperature > 0:
                probs = F.softmax(logit_step / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            else:
                next_tok = logit_step.argmax(dim=-1)  # [B]

            # Replace tokens from already-finished samples with padding sentinel 0.
            next_tok = torch.where(all_eos, torch.zeros_like(next_tok), next_tok)

            lp = F.log_softmax(logit_step, dim=-1)
            tok_lp = lp[torch.arange(bsz, device=device), next_tok]
            tok_lp = torch.where(all_eos, torch.zeros_like(tok_lp), tok_lp)

            gen_tokens[:, step] = next_tok
            gen_log_probs[:, step] = tok_lp
            gen_mask[:, step] = ~all_eos

            all_eos = all_eos | (next_tok == _PALIGEMMA_EOS_TOKEN_ID)
            if all_eos.all():
                break

            nxt_emb = self._embed_language_tokens(next_tok[:, None])  # [B, 1, D]

            # Each generated token uses causal attention: it attends to all
            # valid prefix positions plus all previously generated positions.
            step_att = torch.zeros(bsz, 1, kv_len, dtype=torch.bool, device=device)
            for b in range(bsz):
                if not all_eos[b]:
                    step_att[b, 0, : valid_counts[b]] = True
                    if step > 0:
                        step_att[b, 0, prefix_len : prefix_len + step] = True
                    step_att[b, 0, prefix_len + step] = True

            att_4d_step = torch.where(
                step_att[:, :, None, :], 0.0, torch.finfo(torch.float32).min
            ).float()  # [B, 1, 1, kv_len]

            step_pos_ids = cur_pos_ids[:, None]  # [B, 1]
            cur_pos_ids = cur_pos_ids + 1

            step_out = self.paligemma.language_model(
                inputs_embeds=nxt_emb,
                attention_mask=att_4d_step,
                position_ids=step_pos_ids,
                past_key_values=kv_cache,
                use_cache=True,
            )
            next_logit = self.paligemma.language_model.lm_head(
                step_out.last_hidden_state.to(torch.bfloat16)
            ).float()  # [B, 1, V]
            kv_cache = step_out.past_key_values

        # Decode generated tokens to continuous actions via FAST tokenizer.
        actions_list: list[np.ndarray] = []
        gen_tokens_cpu = gen_tokens.cpu().numpy()

        for b in range(bsz):
            toks_b = gen_tokens_cpu[b]
            # Trim at first zero (EOS/padding sentinel).
            nonzero_mask = toks_b != 0
            end = int(nonzero_mask.sum())
            valid_toks = toks_b[:end] if end > 0 else toks_b

            try:
                act = self._fast_tokenizer.extract_actions(
                    valid_toks,
                    action_horizon=self.config.action_chunk,
                    action_dim=self.config.action_env_dim,
                )  # [action_chunk, action_env_dim]
            except (ValueError, IndexError, RuntimeError):
                self.logger.warning(
                    f"FASTTokenizer.extract_actions failed for batch item {b}; "
                    "returning zero actions."
                )
                act = np.zeros(
                    (self.config.action_chunk, self.config.action_env_dim),
                    dtype=np.float32,
                )
            actions_list.append(act)

        actions_np = np.stack(actions_list, axis=0)  # [B, action_chunk, action_env_dim]
        actions = torch.from_numpy(actions_np).float().to(device)

        return {
            "actions": actions,
            "generated_tokens": gen_tokens,   # [B, max_new_tokens]
            "token_log_probs": gen_log_probs,  # [B, max_new_tokens]
            "token_mask": gen_mask,            # [B, max_new_tokens] bool
        }

    # ------------------------------------------------------------------
    # RL: recompute log-probs for stored action tokens
    # ------------------------------------------------------------------

    def compute_action_logprobs(
        self,
        observation: "_model.Observation",
        action_tokens: torch.Tensor,
        action_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Recompute per-token log-probs for RL training.

        Reconstructs the full token sequence ``[prefix, action_tokens]``,
        runs a single teacher-forcing forward pass, and extracts log-probs
        at the positions that predict each action token.

        Args:
            observation: Preprocessed observation (prefix context only).
            action_tokens: Generated PaliGemma token IDs ``[B, max_new_tokens]``.
            action_token_mask: True for valid (non-padding) positions
                ``[B, max_new_tokens]``.

        Returns:
            token_log_probs: ``[B, max_new_tokens]`` float32, zero for padding.
            values:          ``[B, 1]`` float32 or ``None`` if no value head.
        """
        device = next(self.parameters()).device
        bsz = action_tokens.shape[0]
        max_new_tokens = action_tokens.shape[1]

        images = list(observation.images.values())[: self.config.num_images_in_input]
        img_masks = list(observation.image_masks.values())[: self.config.num_images_in_input]

        prefix_tokens = observation.tokenized_prompt  # [B, T]
        prefix_mask = observation.tokenized_prompt_mask  # [B, T]
        T = prefix_tokens.shape[1]

        # Concatenate prefix and action token sequences.
        all_text_tokens = torch.cat(
            [prefix_tokens, action_tokens.to(prefix_tokens.device)], dim=1
        )  # [B, T + K]
        all_text_mask = torch.cat(
            [prefix_mask.bool(), action_token_mask.to(prefix_mask.device)], dim=1
        )  # [B, T + K]

        # AR mask: prefix positions = 0 (bidirectional), action positions = 1 (causal).
        prefix_ar = torch.zeros(bsz, T, dtype=torch.long, device=device)
        action_ar = action_token_mask.long().to(device)
        all_text_ar = torch.cat([prefix_ar, action_ar], dim=1)

        embs, pad_masks, att_masks = self._embed_prefix(
            images,
            img_masks,
            all_text_tokens.to(device),
            all_text_mask.to(device),
            all_text_ar,
        )
        S = embs.shape[1]
        num_img_tokens = S - all_text_tokens.shape[1]

        att_4d = self._prepare_attn_mask_4d(pad_masks[:, :-1], att_masks[:, :-1])
        position_ids = (torch.cumsum(pad_masks[:, :-1].long(), dim=1) - 1).clamp(min=0)

        hidden = self.paligemma.language_model(
            inputs_embeds=embs[:, :-1],
            attention_mask=att_4d,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state  # [B, S-1, D]

        # hidden[:, num_img_tokens + T - 1 + k, :] predicts action_tokens[:, k]
        # for k in 0 .. K-1.
        action_start = num_img_tokens + T - 1
        action_end = min(action_start + max_new_tokens, S - 1)
        actual_k = action_end - action_start

        action_hidden = hidden[:, action_start:action_end, :]  # [B, actual_k, D]
        action_logits = self.paligemma.language_model.lm_head(
            action_hidden.to(torch.bfloat16)
        ).float()  # [B, actual_k, V]

        target_tokens = action_tokens[:, :actual_k].to(device)
        log_probs = F.log_softmax(action_logits, dim=-1)
        tok_lp = log_probs.gather(2, target_tokens[:, :, None].long()).squeeze(-1)  # [B, actual_k]

        if actual_k < max_new_tokens:
            pad = torch.zeros(bsz, max_new_tokens - actual_k, dtype=tok_lp.dtype, device=device)
            tok_lp = torch.cat([tok_lp, pad], dim=1)

        # Zero-out padding positions.
        tok_lp = tok_lp * action_token_mask.float().to(device)

        # Optional value head: use hidden state at the last valid prefix position.
        values = None
        if self.config.add_value_head and hasattr(self, "value_head"):
            valid_cnts = pad_masks[:, : num_img_tokens + T].long().sum(dim=1)
            last_valid = (valid_cnts - 1).clamp(min=0)
            val_h = hidden[torch.arange(bsz, device=device), last_valid]
            values = self.value_head(val_h.float()).squeeze(-1)[:, None]  # [B, 1]

        return tok_lp, values


class OpenPiFastForRLActionPrediction(OpenPiFastPytorch, BasePolicy):
    """Pi0-Fast wrapped for RLinf's embodied RL training loop.

    Mirrors the interface of ``OpenPi0ForRLActionPrediction`` so it can be
    plugged into the same rollout and actor workers with minimal changes.
    """

    @property
    def _no_split_modules(self) -> list[str]:
        """FSDP layer classes that should not be split across devices."""
        return [
            "SiglipEncoderLayer",
            "GemmaDecoderLayer",
            "SiglipVisionEmbeddings",
            "GemmaRMSNorm",
            "GemmaRotaryEmbedding",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        """FSDP module name substrings that should not be split."""
        return ["lm_head", "value_head"]

    def __init__(self, config: OpenPiFastConfig) -> None:
        OpenPiFastPytorch.__init__(self, config)
        self.logger = get_logger()
        self.global_step = 0
        self._input_transform: Any = None
        self._output_transform: Any = None

        for name, module in self.named_modules():
            parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", parts[-1] if parts else name)

    def set_global_step(self, step: int) -> None:
        """Update the global training step counter."""
        self.global_step = step

    # ------------------------------------------------------------------
    # Transform wiring
    # ------------------------------------------------------------------

    def setup_wrappers(
        self,
        transforms: Sequence[Any] = (),
        output_transforms: Sequence[Any] = (),
    ) -> None:
        """Compose and store input/output transform pipelines.

        Args:
            transforms: Sequence of input data transforms.
            output_transforms: Sequence of output data transforms.
        """
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(
        self, obs: dict[str, Any], transpose: bool = True
    ) -> dict[str, Any]:
        """Apply the input transform pipeline to a raw observation dict.

        Args:
            obs: Raw observation dict (may contain a ``"prompt"`` key on the
                first pass, or only ``"/"``-keyed sensor fields on the second).
            transpose: If True, convert images from ``[C, H, W]`` to
                ``[H, W, C]`` before passing to the transform.

        Returns:
            Transformed observation dict with tensor values.
        """
        inputs = jax.tree.map(lambda x: x, obs)
        first_process = "prompt" in inputs
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {k: v for k, v in inputs.items() if "/" in k}

        # Tensor → numpy before handing to jax/openpi transforms.
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x,
            inputs,
        )
        bsz = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        transformed: list[dict] = []
        for i in range(bsz):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0) if len(x.shape) == 3 else x,
                    sample,
                )
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = _PLACEHOLDER_PROMPT
            transformed.append(self._input_transform(sample))

        result = jax.tree.map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()), *transformed
        )
        if not first_process:
            result["tokenized_prompt"] = obs["tokenized_prompt"]
            result["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return result

    def output_transform(self, outputs: dict[str, Any]) -> dict[str, Any]:
        """Apply the output transform pipeline (e.g. unnormalise actions).

        Args:
            outputs: Dict containing at least an ``"actions"`` tensor.

        Returns:
            Transformed output dict with environment-scale actions.
        """
        bsz = outputs["actions"].shape[0]
        transformed: list[dict] = []
        for i in range(bsz):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            transformed.append(self._output_transform(sample))
        result = jax.tree.map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()), *transformed
        )
        result["actions"] = result["actions"][:, : self.config.action_chunk]
        return result

    def precision_processor(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Move all tensors in an observation dict to the model device.

        Args:
            obs: Observation dict (possibly nested) with tensor values.

        Returns:
            Observation dict with all tensors on the model device.
        """
        device = next(self.parameters()).device
        for k, v in obs.items():
            if isinstance(v, list):
                obs[k] = [
                    x.to(device).contiguous() if torch.is_tensor(x) else x for x in v
                ]
            elif torch.is_tensor(v):
                obs[k] = v.to(device).contiguous()
        return obs

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def freeze_vlm(self) -> None:
        """Freeze PaliGemma parameters so only added heads are trained."""
        if self.config.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False
            self.logger.info("[Pi0-Fast] PaliGemma frozen (train_expert_only=True).")

    # ------------------------------------------------------------------
    # Forward dispatch (BasePolicy interface)
    # ------------------------------------------------------------------

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any) -> Any:
        """Dispatch to the appropriate forward method based on ``forward_type``.

        Args:
            forward_type: One of ``ForwardType.DEFAULT`` or ``ForwardType.SFT``.
            **kwargs: Forwarded to the selected method.

        Returns:
            Output of the dispatched method.

        Raises:
            NotImplementedError: For unsupported ``forward_type`` values.
        """
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError(
                f"forward_type={forward_type} is not supported for Pi0-Fast."
            )

    def sft_forward(self, data: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        """SFT training: teacher-forcing cross-entropy loss.

        Args:
            data: Batch dict with ``"observation"`` and ``"actions"`` keys.
            **kwargs: Ignored; kept for API compatibility.

        Returns:
            Scalar CE loss tensor.
        """
        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()
        observation = data["observation"]
        actions = data["actions"]
        return OpenPiFastPytorch.forward(self, observation, actions)

    def default_forward(
        self,
        forward_inputs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """RL forward: recompute log-probs for stored action tokens.

        Reconstructs the full token sequence from ``forward_inputs``, runs a
        teacher-forcing forward pass, and returns per-token log-probs compatible
        with the PPO actor worker interface.

        Args:
            forward_inputs: Replay-buffer sample dict containing (among other
                keys) ``"action_tokens"``, ``"action_token_mask"``,
                ``"tokenized_prompt"``, and ``"tokenized_prompt_mask"``.
            **kwargs: May contain ``"compute_values"`` (bool).

        Returns:
            Dict with keys:

            - ``logprobs``:  ``[B, max_new_tokens]`` float32.
            - ``values``:    ``[B, 1]`` float32 (zeros if no value head).
            - ``entropy``:   ``[B, 1]`` float32 (zeros; entropy not tracked).
        """
        device = next(self.parameters()).device

        processed_obs = self.input_transform(forward_inputs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        action_tokens = forward_inputs["action_tokens"].to(device)       # [B, K]
        action_token_mask = forward_inputs["action_token_mask"].to(device)  # [B, K]

        token_log_probs, values = self.compute_action_logprobs(
            observation, action_tokens, action_token_mask
        )

        bsz = token_log_probs.shape[0]
        if values is None:
            values = torch.zeros(bsz, 1, dtype=torch.float32, device=device)

        entropy = torch.zeros_like(values)

        return {
            "logprobs": token_log_probs,  # [B, max_new_tokens]
            "values": values,             # [B, 1]
            "entropy": entropy,           # [B, 1]
        }

    # ------------------------------------------------------------------
    # Rollout interface (BasePolicy)
    # ------------------------------------------------------------------

    def obs_processor(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert raw environment observation to policy input format.

        Args:
            env_obs: Raw environment observation dict.

        Returns:
            Policy-format observation dict.
        """
        processed: dict[str, Any] = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "states" in env_obs:
            processed["observation/state"] = env_obs["states"]
        if env_obs.get("wrist_images") is not None:
            processed["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            processed["observation/extra_view_image"] = env_obs["extra_view_images"]
        return processed

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Full rollout step: transform obs → generate actions → store replay data.

        Args:
            env_obs: Raw environment observation dict.
            mode: ``"train"`` or ``"eval"``; unused for Pi0-Fast but kept for
                API consistency.
            **kwargs: Ignored.

        Returns:
            Tuple of:

            - ``actions_env``:  Environment-scale actions ``[B, action_chunk, action_env_dim]``.
            - result dict with keys ``prev_logprobs``, ``prev_values``, ``forward_inputs``.
        """
        raw_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(raw_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        with torch.no_grad():
            out = self.sample_actions(observation)

        actions_env = self.output_transform(
            {"actions": out["actions"], "state": observation.state}
        )["actions"]

        # Build forward_inputs for the RL replay buffer.
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in raw_obs.items() if k != "prompt"}
        )
        forward_inputs: dict[str, Any] = {
            "action_tokens": out["generated_tokens"].cpu(),   # [B, max_new_tokens]
            "action_token_mask": out["token_mask"].cpu(),      # [B, max_new_tokens]
            "tokenized_prompt": processed_obs["tokenized_prompt"].cpu(),
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"].cpu(),
            "action": actions_env.reshape(actions_env.shape[0], -1).contiguous().cpu(),
            "model_action": out["actions"].reshape(out["actions"].shape[0], -1).contiguous().cpu(),
        }
        if "token_ar_mask" in processed_obs:
            forward_inputs["token_ar_mask"] = processed_obs["token_ar_mask"].cpu()
        forward_inputs.update(cloned_obs)

        prev_logprobs = out["token_log_probs"]  # [B, max_new_tokens]
        prev_values = torch.zeros(prev_logprobs.shape[0], 1, dtype=torch.float32)

        return actions_env, {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
