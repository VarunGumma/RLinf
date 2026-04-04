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
and the ground-truth actions, it runs the VLA in inference mode to predict
actions, then computes the **MSE** between the predicted actions and the
ground-truth actions.  This per-sample MSE serves as the loss that is
converted into rewards (via the VLALossReward transform) for GRPO training
of the VLM.

Note: the reward is *not* the VLA training loss (logprobs / cross-entropy),
but the action-space MSE.  For autoregressive VLA models (e.g. pi0-fast)
the predicted tokens are detokenised back to continuous actions before the
MSE is computed.
"""

import copy
import os

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards.vla import VLALossReward
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Worker
from rlinf.utils.utils import clear_memory


class VLARewardWorker(Worker):
    """Frozen VLA model that scores VLM outputs via action-space MSE.

    During the VLM-training phase (GRPO) the VLM generates K candidate text
    outputs per prompt.  For each candidate the frozen VLA predicts actions
    given the candidate text and the current observation.  The per-sample
    **MSE** between the predicted actions and the ground-truth actions is the
    loss signal; it is then converted to a reward (lower MSE → higher reward)
    by :class:`VLALossReward`.

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
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self.device = torch.cuda.current_device()

        # Build VLA model from the vla_reward config section
        vla_model_cfg = copy.deepcopy(self.cfg.vla_reward.model)
        self.vla_model: BasePolicy = get_model(vla_model_cfg)
        self.vla_model.eval()

        # When offload is enabled the model lives on CPU between calls;
        # otherwise it stays on GPU permanently.
        if not self.enable_offload:
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
        """Read VLM outputs + observations from *input_channel*, compute
        action-space MSE rewards, and write them to *output_channel*.

        Expected input (dict on channel)::

            {
                "env_obs": dict,  # raw observations with VLM text as
                # task_descriptions.  Keys:
                #   "main_images": [K, H, W, C]
                #   "states": [K, state_dim]
                #   "task_descriptions": list[str] (len K)
                #   (optional) "wrist_images", "extra_view_images"
                "ground_truth_actions": Tensor,  # [K, action_chunk, action_dim]
            }

        Output (dict on channel)::

            {
                "rewards": Tensor,  # [K]  scalar rewards per sample
                "vla_losses": Tensor,  # [K]  raw per-sample MSE values
            }
        """
        batch = input_channel.get()

        env_obs = batch["env_obs"]
        ground_truth_actions = batch["ground_truth_actions"]

        vla_losses = self._compute_action_mse(env_obs, ground_truth_actions)
        rewards = self.reward_fn(vla_losses)

        output_channel.put(
            {"rewards": rewards.cpu(), "vla_losses": vla_losses.cpu()},
            async_op=True,
        )

    def compute_vla_rewards_direct(
        self,
        env_obs: dict,
        ground_truth_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Synchronous variant that returns rewards directly (no channels).

        This is useful when the runner orchestrates communication itself rather
        than relying on channel-based messaging.

        Args:
            env_obs: Raw observations dict with VLM text already set in
                ``task_descriptions``.  Same format as the ``"env_obs"``
                field described in :meth:`compute_vla_rewards`.
            ground_truth_actions: Ground-truth actions ``[K, action_chunk, action_dim]``.

        Returns:
            ``(rewards, vla_losses)`` both of shape ``[K]``.
        """
        if self.enable_offload:
            self.vla_model.to(self.device)

        vla_losses = self._compute_action_mse(env_obs, ground_truth_actions)
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
    def _compute_action_mse(
        self,
        env_obs: dict,
        ground_truth_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Run frozen VLA inference and return per-sample action MSE.

        The frozen VLA predicts actions from ``env_obs`` (which already
        contains the VLM-generated text in ``task_descriptions``).  The MSE
        between the predicted actions and ``ground_truth_actions`` is
        returned.  For autoregressive VLA models (e.g. pi0-fast) the
        predicted action tokens are automatically detokenised back to
        continuous space by the model's ``output_transform``.

        Args:
            env_obs: Raw observation dict.  Must contain at least
                ``main_images``, ``states``, and ``task_descriptions``.
            ground_truth_actions: Ground-truth actions of shape
                ``[K, action_chunk, action_dim]``.

        Returns:
            Per-sample MSE of shape ``[K]``.
        """
        # Move observations to device
        env_obs_device = {}
        for k, v in env_obs.items():
            if isinstance(v, torch.Tensor):
                env_obs_device[k] = v.to(self.device)
            else:
                env_obs_device[k] = v
        ground_truth_actions = ground_truth_actions.to(self.device)

        with torch.no_grad():
            # predict_action_batch runs full inference (denoising for
            # flow-matching, autoregressive decoding for token-based) and
            # applies output_transform, returning actions in env space.
            predicted_actions, _ = self.vla_model.predict_action_batch(
                env_obs=env_obs_device,
                mode="eval",
                compute_values=False,
            )

        # predicted_actions: [K, action_chunk, action_dim] (env-space)
        # Reshape ground truth to match if necessary
        if predicted_actions.shape != ground_truth_actions.shape:
            ground_truth_actions = ground_truth_actions.reshape(predicted_actions.shape)

        # Per-sample MSE: average over action_chunk and action_dim
        per_sample_mse = (
            (predicted_actions - ground_truth_actions)
            .pow(2)
            .mean(dim=tuple(range(1, predicted_actions.dim())))
        )

        return per_sample_mse
