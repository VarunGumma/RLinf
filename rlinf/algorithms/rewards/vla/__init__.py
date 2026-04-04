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

"""VLA loss-as-reward for GRPO-based VLM training in the agentic VLM+VLA setup.

When training the VLM with GRPO, the frozen VLA's loss on each VLM output
serves as the reward signal.  Lower VLA loss means the VLM produced a better
instruction for the VLA, so the reward is the negated loss.
"""

import torch


class VLALossReward:
    """Converts VLA loss values into scalar rewards for GRPO.

    Supported reward transforms:

    * ``"negate"`` (default): ``reward = -loss``
    * ``"exp_negate"``: ``reward = exp(-loss)``  (bounded in (0, 1])

    Args:
        cfg: Reward config (DictConfig or dict-like) with optional keys:
            - ``reward_transform``: one of ``"negate"``, ``"exp_negate"``
            - ``scale``: multiplicative scale applied *after* the transform
    """

    def __init__(self, cfg=None):
        self.transform = "negate"
        self.scale = 1.0
        if cfg is not None:
            self.transform = getattr(cfg, "reward_transform", "negate")
            self.scale = float(getattr(cfg, "reward_scale", 1.0))

    def __call__(
        self,
        vla_losses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rewards from VLA losses.

        Args:
            vla_losses: Tensor of shape ``[K]`` containing per-sample VLA
                losses (one per VLM rollout output).

        Returns:
            Rewards tensor of the same shape.
        """
        if self.transform == "negate":
            rewards = -vla_losses
        elif self.transform == "exp_negate":
            rewards = torch.exp(-vla_losses)
        else:
            raise ValueError(
                f"Unknown reward transform '{self.transform}'. "
                "Supported: 'negate', 'exp_negate'."
            )
        return rewards * self.scale
