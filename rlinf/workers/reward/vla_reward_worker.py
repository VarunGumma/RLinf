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

"""VLA Reward Worker for the agentic VLM + VLA setup.

This worker holds a **frozen** VLA (OpenPI-based pi0/pi0.5/pi0-fast) model.
Given a batch of VLM-generated text outputs, the corresponding observations,
and the ground-truth actions, it runs the VLA forward pass to compute the
flow-matching / CE loss for each sample, and converts those losses into
rewards (via the VLALossReward transform) that are used by GRPO to train
the VLM.
"""

import copy
import os
from typing import Optional

import torch
from omegaconf import DictConfig, open_dict

from rlinf.algorithms.rewards.vla import VLALossReward
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils.utils import clear_memory


class VLARewardWorker(Worker):
    """Frozen VLA model that scores VLM outputs by computing the VLA loss.

    The worker is designed to be launched as a Ray worker group and called
    from the :class:`AgenticVLMVLARunner` during the VLM-training phase.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg
        self.enable_offload = cfg.vla_reward.get("enable_offload", False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def init_worker(self) -> None:
        """Load the frozen VLA model and set up the reward transform."""
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        # Build VLA model from the vla_reward config section
        vla_model_cfg = copy.deepcopy(self.cfg.vla_reward.model)
        self.vla_model: BasePolicy = get_model(vla_model_cfg)
        self.vla_model.eval()
        self.vla_model.to(self.device)

        # Freeze all parameters
        for param in self.vla_model.parameters():
            param.requires_grad = False

        # Reward transform (negated loss → reward)
        self.reward_fn = VLALossReward(self.cfg.vla_reward.get("reward", None))

        self.log_info("VLARewardWorker initialized with frozen VLA model.")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    @Worker.timer("compute_vla_rewards")
    def compute_vla_rewards(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Read VLM outputs + observations from *input_channel*, compute VLA
        loss-based rewards, and write them to *output_channel*.

        Expected input (dict on channel)::

            {
                "vlm_texts": list[str],         # K VLM-generated text outputs
                "instructions": list[str],       # original task instructions (len K)
                "forward_inputs": dict,           # batched VLA forward_inputs (B=K)
                "ground_truth_actions": Tensor,  # [K, action_chunk, action_dim]
            }

        Output (dict on channel)::

            {
                "rewards": Tensor,   # [K]  scalar rewards per sample
                "vla_losses": Tensor, # [K]  raw VLA losses per sample
            }
        """
        batch = input_channel.get()

        forward_inputs = batch["forward_inputs"]
        ground_truth_actions = batch["ground_truth_actions"]

        vla_losses = self._compute_vla_losses(forward_inputs, ground_truth_actions)
        rewards = self.reward_fn(vla_losses)

        output_channel.put(
            {"rewards": rewards.cpu(), "vla_losses": vla_losses.cpu()},
            async_op=True,
        )

    def compute_vla_rewards_direct(
        self,
        forward_inputs: dict,
        ground_truth_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synchronous variant that returns rewards directly (no channels).

        This is useful when the runner orchestrates communication itself rather
        than relying on channel-based messaging.

        Returns:
            ``(rewards, vla_losses)`` both of shape ``[K]``.
        """
        if self.enable_offload:
            self.vla_model.to(self.device)

        vla_losses = self._compute_vla_losses(forward_inputs, ground_truth_actions)
        rewards = self.reward_fn(vla_losses)

        if self.enable_offload:
            self.vla_model.to("cpu")
            clear_memory(sync=False)

        return rewards, vla_losses

    # ------------------------------------------------------------------
    # Weight sync (called when switching from VLA-train → VLM-train)
    # ------------------------------------------------------------------
    def update_vla_weights(self, state_dict: dict) -> None:
        """Replace the frozen VLA weights (e.g. after a VLA training phase)."""
        self.vla_model.load_state_dict(state_dict, strict=False)
        self.vla_model.eval()
        self.log_info("VLA reward worker: weights updated.")

    async def recv_vla_weights(self, src_group_name: str, src_rank: int = 0) -> None:
        """Receive updated VLA weights from the VLA actor worker group."""
        state_dict = await self.recv(
            src_group_name=src_group_name,
            src_rank=src_rank,
            async_op=True,
        ).async_wait()
        self.update_vla_weights(state_dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_vla_losses(
        self,
        forward_inputs: dict,
        ground_truth_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen VLA forward and return per-sample losses.

        For flow-matching models (pi0, pi0.5) the default_forward returns
        ``{"logprobs": ...}`` and the loss is ``-logprobs.mean(dim=-1)``.
        For autoregressive models (pi0-fast) the loss is cross-entropy.

        Returns:
            Losses of shape ``[K]``.
        """
        # Move inputs to device
        forward_inputs_device = {}
        for k, v in forward_inputs.items():
            if isinstance(v, torch.Tensor):
                forward_inputs_device[k] = v.to(self.device)
            else:
                forward_inputs_device[k] = v
        ground_truth_actions = ground_truth_actions.to(self.device)

        with torch.no_grad():
            output = self.vla_model(
                forward_inputs=forward_inputs_device,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
            )

        # Compute per-sample loss from logprobs
        logprobs = output["logprobs"]  # [K, action_chunk, action_dim] or [K, seq_len]
        # Average over the non-batch dimensions to get per-sample scalar
        if logprobs.dim() > 1:
            per_sample_loss = -logprobs.reshape(logprobs.shape[0], -1).mean(dim=-1)
        else:
            per_sample_loss = -logprobs

        return per_sample_loss
