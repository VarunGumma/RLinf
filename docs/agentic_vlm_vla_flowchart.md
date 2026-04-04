# Agentic VLM + VLA — Architecture Flowchart

This document describes the training architecture for the `agentic_vlm_vla`
task type, which alternates between supervised VLA training (Phase A) and
GRPO-based VLM training (Phase B) with independently configurable phase
durations.

---

## System overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       AgenticVLMVLARunner                               │
│                                                                         │
│   global_step  ──►  phase switching (decoupled intervals)               │
│                     ┌──────────────────┐   ┌──────────────────┐         │
│                     │     Phase A      │   │     Phase B      │         │
│                     │  (VLA, N_vla     │◄─►│  (VLM, N_vlm     │         │
│                     │   steps)         │   │   steps)         │         │
│                     └──────────────────┘   └──────────────────┘         │
│                                                                         │
│   On switch VLA→VLM:  sync VLA weights → VLARewardWorker                │
│                                                                         │
│   VLM training supports:                                                │
│     • Standard GRPO  (vlm_adv_type: grpo)                               │
│     • Dr. GRPO       (vlm_adv_type: dr_grpo) — length-bias corrected    │
│     • Optional KL penalty against reference policy (vlm_kl_coeff > 0)   │
└─────────────────────────────────────────────────────────────────────────┘
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

## Phase B — VLM GRPO / Dr. GRPO Training

The VLA is **frozen**. Rollouts are generated only in this phase.

```
┌──────────────────────────────────────────────────────────────────┐
│  Phase B  (VLM trains via GRPO / Dr. GRPO, VLA frozen)           │
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
│  Step 3 — Advantages (GRPO or Dr. GRPO)                          │
│  ┌────────────────────────────────────────────┐                  │
│  │  calculate_adv_and_returns(                │                  │
│  │    task_type="reasoning",                  │                  │
│  │    adv_type="grpo" | "dr_grpo",            │                  │
│  │    rewards=rewards,                        │                  │
│  │    group_size=K                            │                  │
│  │  )                                         │                  │
│  │                                            │                  │
│  │  GRPO:    z-score within group             │                  │
│  │  Dr.GRPO: z-score + group-mean-length      │                  │
│  │           scaling (corrects verbosity bias) │                  │
│  │  → advantages, returns                     │                  │
│  └────────────────┬───────────────────────────┘                  │
│                   │                                              │
│                   ▼                                              │
│  Step 4 — VLM policy-gradient update (+ optional KL penalty)     │
│  ┌────────────────────────────────────────────┐                  │
│  │  VLM Actor (FSDPActor)                     │                  │
│  │  run_vlm_grpo_training(                    │                  │
│  │    advantages=advantages,                  │                  │
│  │    rewards=rewards,                        │                  │
│  │    vlm_kl_coeff=β,   ← KL weight          │                  │
│  │    vlm_kl_type="low_var_kl"                │                  │
│  │  )                                         │                  │
│  │                                            │                  │
│  │  loss = policy_gradient_loss               │                  │
│  │       + β × KL(π_θ ∥ π_ref)   (if β > 0)  │                  │
│  └────────────────────────────────────────────┘                  │
│                                                                  │
│  Output: vlm/mean_reward, vlm/mean_advantage, vlm_train/*        │
└──────────────────────────────────────────────────────────────────┘
```

**Key points:**
- The reward is the **MSE between predicted and ground-truth actions** in
  continuous space — it is *not* the VLA training loss (logprobs /
  cross-entropy). For pi0-fast the VLA automatically detokenises predicted
  action tokens back to continuous values via `output_transform` before the MSE
  is computed.
- **Dr. GRPO** scales every token in the group by the group's mean response
  length instead of each response's individual length, preventing longer outputs
  from receiving disproportionate gradient (arXiv 2503.20783).
- **KL penalty** (`vlm_kl_coeff > 0`) constrains the VLM policy to stay close
  to a reference policy, preventing reward hacking. Supports `"kl"`, `"abs"`,
  `"mse"`, and `"low_var_kl"` penalty types.

---

## Phase switching and weight sync

```
  Phase A (VLA trains)                       Phase B (VLM trains via GRPO)
 ┌────────────────────────┐                 ┌─────────────────────────────┐
 │  N_vla steps of VLA    │  ── switch ──►  │  N_vlm steps of VLM GRPO   │
 │  supervised (SFT)      │                 │  (or Dr. GRPO) with frozen  │
 │  training              │  ◄── switch ──  │  VLA rewards + optional KL  │
 └────────────────────────┘                 └─────────────────────────────┘
         │                                            ▲
         │  On VLA→VLM transition:                    │
         │    VLA actor state_dict ─────────────────►  │
         │    loaded into VLARewardWorker              │
         │    (so reward uses latest VLA weights)      │
         └────────────────────────────────────────────┘
```

Phase intervals are **independently configurable**:
- `runner.vla_alternating_interval` — steps per VLA block (default: 10).
- `runner.vlm_alternating_interval` — steps per VLM block (default: 10).
- The legacy `runner.alternating_interval` is accepted as a fallback for both.

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

## Dr. GRPO — length-bias correction

Standard GRPO normalises each response's gradient by its own token count.
Longer responses thus receive more total gradient, which can incentivise
verbosity.  **Dr. GRPO** fixes this:

```
Standard GRPO (per-response normalisation):
  loss_i = (1 / |o_i|) × Σ_t  advantage_t × log π(t)

Dr. GRPO (group-mean-length normalisation):
  L̄ = mean(|o_1|, |o_2|, …, |o_K|)            ← group mean length
  loss_i = (L̄ / |o_i|) × (1 / L̄) × Σ_t  advantage_t × log π(t)
         ≡  (1 / |o_i|) × scale_i × Σ_t  advantage_t × log π(t)
  where scale_i = L̄ / |o_i|
```

Every token in the group is scaled by `group_mean_len / response_len`, so
each response contributes equally regardless of its verbosity.

Set `algorithm.vlm_adv_type: dr_grpo` in config to enable.

Reference: *"Understanding R1-Zero-Like Training: A Critical Perspective"*
(arXiv 2503.20783).

---

## KL divergence penalty

An optional KL divergence penalty constrains the VLM policy to stay close to
a reference policy during Phase B training:

```
total_loss = GRPO_policy_gradient_loss + β × KL(π_θ ∥ π_ref)
```

where `β = vlm_kl_coeff` (default `0.0` = disabled).

Supported penalty types (`vlm_kl_type`):

| Type            | Formula                                          |
|-----------------|--------------------------------------------------|
| `"kl"` / `"k1"` | log π_θ − log π_ref                              |
| `"abs"`         | |log π_θ − log π_ref|                            |
| `"mse"` / `"k2"`| 0.5 × (log π_θ − log π_ref)²                    |
| `"low_var_kl"` / `"k3"` | clamp(exp(log π_ref − log π_θ) − (log π_ref − log π_θ) − 1) |

The `"low_var_kl"` variant (Schulman, 2020) is the default and provides a
lower-variance estimator of the KL divergence.

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
