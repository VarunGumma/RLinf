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

"""Entry-point for the agentic VLM + VLA alternating GRPO training.

Usage
-----
::

    python examples/agentic_vlm_vla/train_agentic.py \\
        --config-name base_agentic

Worker groups created
---------------------
* **vla_actor**   – EmbodiedFSDPActor for VLA training (phase A)
* **vla_rollout** – MultiStepRolloutWorker for VLA action generation
* **vlm_actor**   – FSDPActor for VLM training (phase B, GRPO)
* **vla_reward**  – VLARewardWorker (frozen VLA that scores VLM outputs)
* **env**         – EnvWorker for embodied environment interaction
"""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.agentic_vlm_vla_runner import AgenticVLMVLARunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor, FSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.vla_reward_worker import VLARewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="base_agentic")
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # ------------------------------------------------------------------ #
    # VLA actor (embodied FSDP — trained in phase A)                      #
    # ------------------------------------------------------------------ #
    vla_actor_placement = component_placement.get_strategy("actor")
    vla_actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=vla_actor_placement
    )

    # ------------------------------------------------------------------ #
    # VLA rollout (HF rollout for action generation)                      #
    # ------------------------------------------------------------------ #
    vla_rollout_placement = component_placement.get_strategy("rollout")
    vla_rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=vla_rollout_placement
    )

    # ------------------------------------------------------------------ #
    # Environment                                                         #
    # ------------------------------------------------------------------ #
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # ------------------------------------------------------------------ #
    # VLA reward (frozen VLA that scores VLM outputs)                     #
    # ------------------------------------------------------------------ #
    vla_reward_placement = component_placement.get_strategy("reward")
    vla_reward_group = VLARewardWorker.create_group(cfg).launch(
        cluster,
        name=cfg.vla_reward.get("group_name", "vla_reward"),
        placement_strategy=vla_reward_placement,
    )

    # ------------------------------------------------------------------ #
    # VLM actor (FSDP — trained in phase B with GRPO)                     #
    # ------------------------------------------------------------------ #
    vlm_actor_placement = component_placement.get_strategy("actor")
    vlm_actor_group = FSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.vlm_actor.get("group_name", "vlm_actor"),
        placement_strategy=vlm_actor_placement,
    )

    # ------------------------------------------------------------------ #
    # Runner                                                              #
    # ------------------------------------------------------------------ #
    runner = AgenticVLMVLARunner(
        cfg=cfg,
        vla_actor=vla_actor_group,
        vla_rollout=vla_rollout_group,
        vlm_actor=vlm_actor_group,
        vla_reward=vla_reward_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
