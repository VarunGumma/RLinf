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

"""Agentic VLM + VLA Runner — alternating GRPO / embodied training.

Training protocol
-----------------
The system contains two models:

* **VLM** (Vision-Language Model) — generates textual instructions / plans.
* **VLA** (Vision-Language-Action model, e.g. pi0 / pi0.5 / pi0-fast via
  OpenPI) — consumes those instructions and produces robot actions.

Training alternates between two phases every ``alternating_interval`` steps:

**Phase A — VLA training** (``phase == "vla"``):
    The VLM is frozen.  A single VLM output per prompt is generated and
    concatenated with the task instruction to form the VLA's text input.
    The VLA is then trained with its native loss (flow-matching / CE) using
    the standard embodied RL loop (env interaction → rollout → actor update).

**Phase B — VLM training** (``phase == "vlm"``):
    The VLA is frozen.  For each prompt the VLM generates *K* rollout
    outputs (``group_size``).  Each output is evaluated by the frozen VLA
    which computes its loss — the negated loss becomes the GRPO reward.
    Advantages are computed within each group of K and the VLM is updated
    via a policy-gradient (GRPO) step.
"""

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
from omegaconf.dictconfig import DictConfig

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor, FSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.vla_reward_worker import VLARewardWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

logger = logging.getLogger(__name__)


class AgenticVLMVLARunner:
    """Orchestrates alternating VLM ↔ VLA training.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    vla_actor : EmbodiedFSDPActor
        Worker group for the VLA (trained in phase A, frozen in phase B).
    vla_rollout : MultiStepRolloutWorker
        HuggingFace rollout worker for VLA action generation.
    vlm_actor : FSDPActor
        Worker group for the VLM (trained in phase B, frozen in phase A).
    vla_reward : VLARewardWorker
        Frozen-VLA evaluator that converts VLA loss into GRPO rewards.
    env : EnvWorker
        Environment worker group for embodied interaction.
    """

    def __init__(
        self,
        cfg: DictConfig,
        vla_actor: "EmbodiedFSDPActor",
        vla_rollout: "MultiStepRolloutWorker",
        vlm_actor: "FSDPActor",
        vla_reward: "VLARewardWorker",
        env: "EnvWorker",
    ):
        self.cfg = cfg
        self.vla_actor = vla_actor
        self.vla_rollout = vla_rollout
        self.vlm_actor = vlm_actor
        self.vla_reward = vla_reward
        self.env = env

        # Phase management
        self.alternating_interval = cfg.runner.alternating_interval
        self.phase: str = cfg.runner.get("initial_phase", "vla")
        self.steps_in_current_phase = 0

        # Channels for embodied VLA training (phase A)
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")

        # Channels for VLM GRPO training (phase B)
        self.vlm_reward_input_channel = Channel.create("VLMRewardInput")
        self.vlm_reward_output_channel = Channel.create("VLMRewardOutput")

        # Step counters
        self.global_step = 0
        self.vla_step = 0
        self.vlm_step = 0

        # Timers & logging
        self.weight_sync_interval = cfg.runner.get("weight_sync_interval", 1)
        self.run_timer = Timer(None)
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.logger_inst = get_logger()
        self.metric_logger = MetricLogger(cfg)

        self.set_max_steps()

        # Async logging (same pattern as EmbodiedRunner)
        self.stop_logging = False
        self.log_queue: queue.Queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def _log_worker(self):
        while not self.stop_logging:
            try:
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.warning("Logging error: %s", e)

    def print_metrics_table_async(
        self, step, total_steps, start_time, metrics, start_step=0
    ):
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs
        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    # ------------------------------------------------------------------
    # Worker initialisation
    # ------------------------------------------------------------------
    def init_workers(self):
        """Initialize all worker groups in order to manage peak memory."""
        # VLA side
        vla_rollout_handle = self.vla_rollout.init_worker()
        env_handle = self.env.init_worker()
        vla_rollout_handle.wait()
        env_handle.wait()
        self.vla_actor.init_worker().wait()

        # VLA reward worker (frozen VLA for scoring VLM outputs)
        self.vla_reward.init_worker().wait()

        # VLM actor
        self.vlm_actor.init_worker().wait()

        # Resume support
        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is not None:
            self.logger_inst.info("Resuming from %s", resume_dir)
            vla_ckpt = os.path.join(resume_dir, "vla_actor")
            vlm_ckpt = os.path.join(resume_dir, "vlm_actor")
            if os.path.exists(vla_ckpt):
                self.vla_actor.load_checkpoint(vla_ckpt).wait()
            if os.path.exists(vlm_ckpt):
                self.vlm_actor.load_checkpoint(vlm_ckpt).wait()
            # Parse global_step from directory name (e.g. ".../global_step_42")
            try:
                step_str = resume_dir.rstrip("/").split("global_step_")[-1]
                self.global_step = int(step_str)
            except (ValueError, IndexError):
                self.logger_inst.warning(
                    "Could not parse global_step from resume_dir '%s'; "
                    "starting from step 0.",
                    resume_dir,
                )

    # ------------------------------------------------------------------
    # Weight synchronisation helpers
    # ------------------------------------------------------------------
    def _sync_vla_weights_to_rollout(self):
        """Push VLA actor weights → VLA rollout worker."""
        rollout_handle: Handle = self.vla_rollout.sync_model_from_actor()
        actor_handle: Handle = self.vla_actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def _sync_vla_weights_to_reward(self):
        """Push latest VLA weights → VLA reward worker (frozen evaluator).

        This is called when transitioning from a VLA phase to a VLM phase so
        the reward worker uses the most recent VLA parameters.
        """
        state_dict_handle: Handle = self.vla_actor.get_model_state_dict()
        state_dict = state_dict_handle.wait()
        # state_dict comes back as a list (one per rank); take rank-0
        if isinstance(state_dict, list):
            state_dict = state_dict[0]
        self.vla_reward.update_vla_weights(state_dict).wait()

    # ------------------------------------------------------------------
    # Phase switching
    # ------------------------------------------------------------------
    def _switch_phase(self):
        old_phase = self.phase
        if self.phase == "vla":
            # Transitioning to VLM training — sync latest VLA to reward worker
            self._sync_vla_weights_to_reward()
            self.phase = "vlm"
        else:
            # Transitioning to VLA training — nothing extra needed because
            # the VLA actor already has the latest weights.
            self.phase = "vla"
        self.steps_in_current_phase = 0
        self.logger_inst.info(
            "Phase switch: %s → %s at global_step=%d",
            old_phase,
            self.phase,
            self.global_step,
        )

    # ------------------------------------------------------------------
    # Phase A — VLA training (embodied RL loop)
    # ------------------------------------------------------------------
    def _run_vla_training_step(self):
        """One step of embodied VLA training (mirrors EmbodiedRunner.run body)."""
        self.vla_actor.set_global_step(self.global_step)
        self.vla_rollout.set_global_step(self.global_step)

        with self.timer("vla/sync_weights"):
            if self.vla_step % self.weight_sync_interval == 0:
                self._sync_vla_weights_to_rollout()

        with self.timer("vla/generate_rollouts"):
            env_handle: Handle = self.env.interact(
                input_channel=self.env_channel,
                rollout_channel=self.rollout_channel,
                reward_channel=None,
                actor_channel=self.actor_channel,
            )
            rollout_handle: Handle = self.vla_rollout.generate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
            self.vla_actor.recv_rollout_trajectories(
                input_channel=self.actor_channel
            ).wait()
            rollout_handle.wait()

        with self.timer("vla/cal_adv"):
            adv_metrics = self.vla_actor.compute_advantages_and_returns().wait()

        with self.timer("vla/train"):
            train_handle: Handle = self.vla_actor.run_training()
            train_metrics = train_handle.wait()

        self.vla_step += 1

        # Collect env results for logging
        env_results = env_handle.wait()
        env_results_list = [r for r in env_results if r is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

        return {
            **{f"vla_rollout/{k}": v for k, v in self._agg(adv_metrics).items()},
            **{f"vla_train/{k}": v for k, v in self._agg(train_metrics).items()},
            **env_metrics,
        }

    # ------------------------------------------------------------------
    # Phase B — VLM training (GRPO with VLA-loss reward)
    # ------------------------------------------------------------------
    def _run_vlm_training_step(self):
        """One step of VLM GRPO training using frozen-VLA loss as reward.

        High-level flow
        ---------------
        1. VLM actor generates K text outputs per prompt (group rollout).
        2. For each output the frozen VLA reward worker computes the VLA
           loss → reward.
        3. GRPO advantages are computed over each group of K rewards.
        4. The VLM actor is updated with the policy-gradient loss.
        """
        group_size = self.cfg.algorithm.group_size

        with self.timer("vlm/generate"):
            # The VLM actor generates rollout outputs.
            # We call the VLM actor's generate_vlm_rollout method which
            # produces K text completions per prompt and stores them together
            # with their log-probs.
            gen_handle: Handle = self.vlm_actor.generate_vlm_rollout(
                group_size=group_size,
                output_channel=self.vlm_reward_input_channel,
            )
            gen_handle.wait()

        with self.timer("vlm/vla_reward"):
            # Evaluate each VLM output through the frozen VLA
            reward_handle: Handle = self.vla_reward.compute_vla_rewards(
                input_channel=self.vlm_reward_input_channel,
                output_channel=self.vlm_reward_output_channel,
            )
            reward_handle.wait()
            reward_data = self.vlm_reward_output_channel.get()
            rewards = reward_data["rewards"]  # [K]

        with self.timer("vlm/advantages"):
            # Compute GRPO advantages within each group.
            # loss_mask shape is (seq_len=1, num_samples=K) because the
            # reasoning-side preprocess expects (seq_len, batch) and all
            # VLM outputs are valid (no padding).
            loss_mask = torch.ones(1, rewards.shape[0])
            advantages, returns = calculate_adv_and_returns(
                task_type="reasoning",
                adv_type=self.cfg.algorithm.get("vlm_adv_type", "grpo"),
                rewards=rewards,
                loss_mask=loss_mask,
                group_size=group_size,
            )

        with self.timer("vlm/train"):
            # Send advantages to VLM actor for training
            train_handle: Handle = self.vlm_actor.run_vlm_grpo_training(
                advantages=advantages,
                rewards=rewards,
            )
            train_metrics = train_handle.wait()

        self.vlm_step += 1

        return {
            "vlm/mean_reward": rewards.mean().item(),
            "vlm/mean_advantage": advantages.mean().item()
            if advantages is not None
            else 0.0,
            **{f"vlm_train/{k}": v for k, v in self._agg(train_metrics).items()},
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        start_step = self.global_step
        start_time = time.time()

        for _step in range(start_step, self.max_steps):
            # Phase switching
            if self.steps_in_current_phase >= self.alternating_interval:
                self._switch_phase()

            with self.timer("step"):
                if self.phase == "vla":
                    step_metrics = self._run_vla_training_step()
                else:
                    step_metrics = self._run_vlm_training_step()

            self.global_step += 1
            self.steps_in_current_phase += 1

            # Checkpointing / eval
            run_val, save_model, is_train_end = check_progress(
                self.global_step,
                self.max_steps,
                self.cfg.runner.get("val_check_interval", -1),
                self.cfg.runner.get("save_interval", self.max_steps),
                1.0,
                run_time_exceeded=False,
            )

            eval_metrics: dict = {}
            if run_val and self.phase == "vla":
                with self.timer("eval"):
                    self._sync_vla_weights_to_rollout()
                    eval_metrics = self._evaluate_vla()
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    self.metric_logger.log(data=eval_metrics, step=_step)

            if save_model:
                self._save_checkpoint()

            # Timing & logging
            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            step_metrics["phase"] = 0.0 if self.phase == "vla" else 1.0
            step_metrics["vla_step"] = float(self.vla_step)
            step_metrics["vlm_step"] = float(self.vlm_step)

            self.metric_logger.log(step_metrics, _step)
            self.metric_logger.log(time_metrics, _step)

            all_metrics = {**step_metrics, **time_metrics, **eval_metrics}
            self.print_metrics_table_async(
                _step, self.max_steps, start_time, all_metrics, start_step
            )

        self.metric_logger.finish()
        self.stop_logging = True
        self.log_queue.join()
        self.log_thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_vla(self) -> dict:
        """Run VLA evaluation in the environment."""
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.vla_rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_list = [r for r in env_results if r is not None]
        return compute_evaluate_metrics(eval_list)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self):
        self.logger_inst.info("Saving checkpoint at step %d.", self.global_step)
        base_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        vla_path = os.path.join(base_dir, "vla_actor")
        vlm_path = os.path.join(base_dir, "vlm_actor")
        os.makedirs(vla_path, exist_ok=True)
        os.makedirs(vlm_path, exist_ok=True)
        self.vla_actor.save_checkpoint(vla_path, self.global_step).wait()
        self.vlm_actor.save_checkpoint(vlm_path, self.global_step).wait()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _agg(metrics_list) -> dict:
        """Aggregate per-rank metrics into means."""
        if not metrics_list:
            return {}
        merged: dict[str, list] = defaultdict(list)
        items = metrics_list if isinstance(metrics_list, list) else [metrics_list]
        for m in items:
            if not m:
                continue
            for k, v in m.items():
                merged[k].append(v)
        return {k: sum(vs) / len(vs) for k, vs in merged.items() if vs}
