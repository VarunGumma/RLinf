# Agentic VLM + VLA — Architecture Flowchart

This document describes the training architecture for the `agentic_vlm_vla`
task type, which alternates between supervised VLA training (Phase A) and
GRPO-based VLM training (Phase B).

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AgenticVLMVLARunner                               │
│                                                                     │
│   global_step  ──►  phase switching every alternating_interval      │
│                     ┌──────────┐      ┌──────────┐                  │
│                     │ Phase A  │ ◄──► │ Phase B  │                  │
│                     │  (VLA)   │      │  (VLM)   │                  │
│                     └──────────┘      └──────────┘                  │
│                                                                     │
│   On switch VLA→VLM:  sync VLA weights → VLARewardWorker            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase A — VLA Supervised Training

The VLM is **frozen**. No environment rollouts occur.

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase A  (VLA trains, VLM frozen)                               │
│                                                                  │
│  ┌──────────┐   1. generate_vlm_rollout(group_size=1)            │
│  │   VLM    │──────────────────────────────────────┐             │
│  │ (frozen) │   produces 1 text per sample         │             │
│  └──────────┘                                      ▼             │
│                                           ┌─────────────────┐    │
│                                           │  Concatenate     │    │
│                                           │  VLM text +      │    │
│                                           │  task instruction │    │
│                                           └────────┬────────┘    │
│                                                    │             │
│  ┌───────────────┐   SFT data                      ▼             │
│  │ Demonstration  │────────────────►  ┌──────────────────────┐   │
│  │ Dataset (obs,  │  ground-truth     │  VLA Actor            │   │
│  │ gt_actions)    │  actions           │  (EmbodiedFSDPActor)  │   │
│  └───────────────┘                    │                      │   │
│                                       │  run_sft_step():     │   │
│                                       │   • native loss      │   │
│                                       │     - pi0/pi0.5:     │   │
│                                       │       flow-match MSE │   │
│                                       │     - pi0-fast: CE   │   │
│                                       │   • optimizer.step() │   │
│                                       └──────────────────────┘   │
│                                                                  │
│  Output: vla_train/actor/sft_loss, grad_norm, lr                 │
└──────────────────────────────────────────────────────────────────┘
```

**Key point:** The VLA is trained with its standard supervised loss — *not*
with RL. The VLM text simply enriches the VLA's text prompt.

---

## Phase B — VLM GRPO Training

The VLA is **frozen**. Rollouts are generated only in this phase.

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase B  (VLM trains via GRPO, VLA frozen)                      │
│                                                                  │
│  Step 1 — VLM rollout                                            │
│  ┌───────────┐   generate_vlm_rollout(group_size=K)              │
│  │   VLM     │─────────────────────────────────────┐             │
│  │ (training)│   produces K text completions        │             │
│  └───────────┘   per prompt (with log-probs)        │             │
│                                                     ▼             │
│  Step 2 — VLA reward scoring                                     │
│  ┌────────────────────────────────────────────────────────┐      │
│  │  VLARewardWorker  (frozen VLA copy)                    │      │
│  │                                                        │      │
│  │  For each of the K VLM outputs:                        │      │
│  │    ┌──────────────────────────────────────────────┐    │      │
│  │    │ 1. Set VLM text as task_description in obs   │    │      │
│  │    │ 2. predict_action_batch(env_obs, mode=eval)  │    │      │
│  │    │    → denoising (pi0/pi0.5) or AR decoding    │    │      │
│  │    │      (pi0-fast) + output_transform            │    │      │
│  │    │    → predicted_actions [K, chunk, dim]        │    │      │
│  │    │ 3. MSE = mean((pred - gt)²)  per sample      │    │      │
│  │    │ 4. reward = transform(MSE)                   │    │      │
│  │    │    "negate":     reward = -MSE               │    │      │
│  │    │    "exp_negate": reward = exp(-MSE)           │    │      │
│  │    └──────────────────────────────────────────────┘    │      │
│  │                                                        │      │
│  │  Output: rewards [K], vla_losses [K]                   │      │
│  └────────────────────────────────────────────────────────┘      │
│                         │                                        │
│                         ▼                                        │
│  Step 3 — GRPO advantages                                        │
│  ┌────────────────────────────────────────────┐                  │
│  │  calculate_adv_and_returns(                │                  │
│  │    task_type="reasoning",                  │                  │
│  │    adv_type="grpo",                        │                  │
│  │    rewards=rewards,                        │                  │
│  │    group_size=K                            │                  │
│  │  )                                         │                  │
│  │  → advantages, returns                     │                  │
│  └────────────────┬───────────────────────────┘                  │
│                   │                                              │
│                   ▼                                              │
│  Step 4 — VLM policy-gradient update                             │
│  ┌────────────────────────────────────────────┐                  │
│  │  VLM Actor (FSDPActor)                     │                  │
│  │  run_vlm_grpo_training(                    │                  │
│  │    advantages=advantages,                  │                  │
│  │    rewards=rewards                         │                  │
│  │  )                                         │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                  │
│  Output: vlm/mean_reward, vlm/mean_advantage, vlm_train/*        │
└──────────────────────────────────────────────────────────────────┘
```

**Key point:** The reward is the **MSE between predicted and ground-truth
actions** in continuous space — it is *not* the VLA training loss (logprobs /
cross-entropy). For pi0-fast the VLA automatically detokenises predicted action
tokens back to continuous values via `output_transform` before the MSE is
computed.

---

## Phase switching and weight sync

```
  Phase A (VLA trains)                    Phase B (VLM trains via GRPO)
 ┌────────────────────┐                  ┌─────────────────────────────┐
 │  N steps of VLA    │  ── switch ──►   │  N steps of VLM GRPO       │
 │  supervised (SFT)  │                  │  with frozen-VLA rewards    │
 │  training          │  ◄── switch ──   │                             │
 └────────────────────┘                  └─────────────────────────────┘
         │                                         ▲
         │  On VLA→VLM transition:                 │
         │    VLA actor state_dict ──────────────►  │
         │    loaded into VLARewardWorker           │
         │    (so reward uses latest VLA weights)   │
         └─────────────────────────────────────────┘
```

The alternating interval is configurable (`runner.alternating_interval`,
default: 10 steps).

---

## Worker groups

| Worker group          | Class                   | Role                                          |
|-----------------------|-------------------------|-----------------------------------------------|
| `vla_actor`           | `EmbodiedFSDPActor`     | VLA model — SFT training in Phase A           |
| `vla_rollout`         | `MultiStepRolloutWorker`| VLA action generation (eval only)             |
| `vlm_actor`           | `FSDPActor`             | VLM model — GRPO training in Phase B          |
| `vla_reward`          | `VLARewardWorker`       | Frozen VLA copy — computes action-MSE rewards |
| `env`                 | `EnvWorker`             | Embodied environment (eval only)              |

---

## Reward computation detail

```
VLM output (text)  ──►  frozen VLA  ──►  predicted_actions [K, chunk, dim]
                                              │
ground_truth_actions [K, chunk, dim]  ────────┤
                                              ▼
                                   MSE = mean((pred - gt)²)
                                              │
                                   ┌──────────┴──────────┐
                                   │  reward_transform   │
                                   │  "negate":  r = -MSE│
                                   │  "exp_negate":      │
                                   │    r = exp(-MSE)    │
                                   └──────────┬──────────┘
                                              │
                                              ▼
                                     reward × scale
```

- **Lower MSE = higher reward** — the VLM is rewarded for producing
  instructions that make the VLA predict actions closer to the ground truth.
- The MSE is computed in **environment action space** (post-`output_transform`),
  ensuring correct comparison even for tokenised models like pi0-fast.

---

## File reference

| File                                                  | Description                                 |
|-------------------------------------------------------|---------------------------------------------|
| `rlinf/runners/agentic_vlm_vla_runner.py`             | Main runner orchestrating both phases       |
| `rlinf/workers/reward/vla_reward_worker.py`           | Frozen-VLA reward worker (action-MSE)       |
| `rlinf/algorithms/rewards/vla/__init__.py`            | `VLALossReward` — MSE → reward transform    |
| `rlinf/workers/actor/fsdp_actor_worker.py`            | `run_sft_step()` for VLA Phase A            |
| `rlinf/config.py`                                     | Model enums + `validate_agentic_vlm_vla_cfg`|
| `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py`     | Liger kernel wiring for new VLMs            |
| `rlinf/algorithms/rewards/__init__.py`                | Reward registry entry for `vla_loss`        |
| `examples/agentic_vlm_vla/train_agentic.py`           | Hydra entry-point script                    |
| `examples/agentic_vlm_vla/config/base_agentic.yaml`   | Reference Hydra config                      |
