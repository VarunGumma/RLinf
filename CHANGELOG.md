# Changelog

All notable changes to the **Agentic VLM + VLA** feature are documented below.

## [Unreleased] — Agentic VLM + VLA Alternating GRPO Training

### Added

#### New task type: `agentic_vlm_vla`
- Registered `"agentic_vlm_vla"` in `SUPPORTED_TASK_TYPE` with a dedicated
  `validate_agentic_vlm_vla_cfg()` config validator (`rlinf/config.py`).

#### New VLM model support
- Added **Qwen3-VL** (`QWEN3_VL`) and **Qwen3-VL-MoE** (`QWEN3_VL_MOE`) to
  the `SupportedModel` enum as `"reasoning"`-category models
  (`rlinf/config.py`).
- Added **Qwen3.5** family support: natively multimodal dense (`QWEN3_5`) and
  MoE (`QWEN3_5_MOE`) to the `SupportedModel` enum (`rlinf/config.py`).
  Unlike Qwen3, Qwen3.5 models handle both vision and text natively, so no
  separate VL or SFT variants are needed.
- Wired Liger-kernel fusions for `QWEN3_VL` and `QWEN3_VL_MOE` in the FSDP
  model manager (`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`).
- Wired Liger-kernel fusions for all Qwen3.5 VL/MoE variants in the FSDP
  model manager (`rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`).

#### Runner — `AgenticVLMVLARunner`
- New runner (`rlinf/runners/agentic_vlm_vla_runner.py`) that orchestrates
  alternating two-phase training:
  - **Phase A (VLA training):** The VLM is frozen; it generates one text output
    per sample that is concatenated with the task instruction. The VLA trains
    with its native supervised loss (flow-matching MSE for pi0/pi0.5,
    cross-entropy for pi0-fast) on demonstration data. **No environment
    rollouts** occur in this phase.
  - **Phase B (VLM training):** The VLA is frozen. The VLM generates *K*
    rollout outputs per prompt (`group_size`). Each output is scored by the
    frozen VLA, which predicts actions and computes the **MSE between predicted
    and ground-truth actions**. The negated MSE is the GRPO reward. Advantages
    are computed per group and the VLM is updated via policy gradient.
- Supports configurable phase durations, checkpoint resume, evaluation, and
  async metric logging.

#### Dr. GRPO advantage variant
- New `"dr_grpo"` advantage type registered in
  `rlinf/algorithms/advantages.py`. Dr. GRPO corrects the token-aggregation
  bias of standard GRPO by normalising every token in a group by the **group
  mean response length** instead of per-response length. This prevents the
  model from being incentivised towards verbosity.
  - Reference: *"Understanding R1-Zero-Like Training: A Critical Perspective"*
    (arXiv 2503.20783).
  - Set `algorithm.vlm_adv_type: dr_grpo` in config to enable.

#### KL divergence penalty for VLM training
- Added an optional **KL divergence penalty** against a reference VLM policy
  during Phase B (VLM GRPO training).
  - Configured via `algorithm.vlm_kl_coeff` (default `0.0` = disabled) and
    `algorithm.vlm_kl_type` (default `"low_var_kl"`; also supports `"kl"`,
    `"abs"`, `"mse"`).
  - The penalty reuses the existing `kl_penalty()` utility from
    `rlinf/algorithms/utils.py` and the KL integration in the FSDPActor
    training loop.
  - The runner passes `vlm_kl_coeff` and `vlm_kl_type` to the VLM actor's
    `run_vlm_grpo_training()`.

#### Decoupled phase alternating intervals
- VLA and VLM training phases can now run for **different** numbers of steps
  before switching:
  - `runner.vla_alternating_interval` — steps per VLA (Phase A) block.
  - `runner.vlm_alternating_interval` — steps per VLM (Phase B) block.
  - The legacy `runner.alternating_interval` is still accepted as a fallback
    when the per-phase keys are absent (backward compatible).
- The runner uses a `_current_phase_interval` property to dispatch the correct
  interval for the active phase.

#### Reward — `VLARewardWorker` and `VLALossReward`
- New reward worker (`rlinf/workers/reward/vla_reward_worker.py`) that holds a
  **frozen copy** of the VLA. For each VLM candidate it:
  1. Calls `predict_action_batch` on the frozen VLA (full denoising /
     autoregressive decoding, including `output_transform` for detokenisation).
  2. Computes **per-sample MSE** between predicted and ground-truth actions.
  3. Converts MSE → reward via `VLALossReward`.
- The reward is specifically the **action-space MSE**, *not* the VLA training
  loss (logprobs / cross-entropy). For autoregressive models such as pi0-fast
  the predicted tokens are detokenised back to continuous actions before MSE
  computation.
- New reward transform class `VLALossReward`
  (`rlinf/algorithms/rewards/vla/__init__.py`) supporting `"negate"` and
  `"exp_negate"` transforms with configurable scale. Registered in the global
  reward registry as `"vla_loss"`.
- `VLARewardWorker` supports GPU offloading, direct (synchronous) reward
  computation, and weight updates at phase transitions.

#### Actor — standalone SFT step
- Added `run_sft_step()` to `EmbodiedFSDPActor`
  (`rlinf/workers/actor/fsdp_actor_worker.py`) — a channel-driven supervised
  training step used during VLA Phase A of the agentic setup.

#### Example config and entry-point
- New Hydra config `examples/agentic_vlm_vla/config/base_agentic.yaml` with
  full documentation of every section (cluster, runner, algorithm, env, VLA
  actor, VLM actor, VLA rollout, VLA reward, KL, Dr. GRPO).
- New entry-point script `examples/agentic_vlm_vla/train_agentic.py`.

### Changed

- **Reward signal (critical fix):** The VLM's GRPO reward is the MSE between
  the VLA's *predicted actions* and the *ground-truth actions* — **not** the
  VLA training loss. Lower MSE → higher reward.
- **VLA Phase A (critical fix):** VLA training is supervised (SFT) with
  VLM-augmented prompts. There are no environment rollouts during VLA training;
  rollouts only occur when the VLM is trained via GRPO.
- Used `isinstance(x, (list, tuple))` instead of `list | tuple` for broader
  Python compatibility in `run_sft_step`.
- **Decoupled intervals (breaking config change):** The single
  `runner.alternating_interval` is superseded by the per-phase
  `runner.vla_alternating_interval` / `runner.vlm_alternating_interval` pair.
  The old key is still read as a fallback.
