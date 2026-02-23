# PANO Research: Complete Technical Documentation

## Overview

This repository contains an empirical investigation into JEPA (Joint-Embedding Predictive Architecture) failure modes in hybrid continuous control. The primary finding is **negative**: standard latent JEPA rollout diverges exponentially in high-dimensional systems under partial observability. As a constructive baseline, PANO (Physics-Anchored Neural Observer) recovers partial task performance via observation-space velocity estimation.

### Two-Line Summary

1. **Standard Latent JEPA rollout fails** — prediction error grows exponentially with rollout steps across all tested environments.
2. **PANO (velocity prediction + Euler integration) recovers partial performance** — by staying in observation space, it avoids latent drift.

---

## Repository Structure

```
pcp-jepa-research/
├── experiments/
│   ├── phase5/               # Bouncing Ball (1D validation)
│   │   ├── f3_jepa.py           # F3-JEPA on 1D bouncing ball
│   │   ├── h2_validation.py     # OOD generalization suite
│   │   ├── figures.py           # Phase 5 figure generation
│   │   └── ...                  # Supporting experiments
│   └── phase6/               # MuJoCo scale-up (main paper results)
│       ├── hopper_pano.py              # PANO vs all baselines
│       ├── hopper_standard_jepa.py     # Standard Latent JEPA (diverges)
│       ├── bulletproof_negative.py     # 3-experiment negative protocol
│       └── neurips_figures.py          # Data-driven figure generation
├── src/
│   ├── evaluation/
│   │   └── stats.py           # Statistical tests (Welch's t, bootstrap CI)
│   ├── environments/          # Environment utilities
│   ├── models/                # Model stubs
│   └── utils/
├── archive_phase1_to_4/       # Archived JAX/Flax experiments (not used)
├── results/                   # Generated JSON results + PDF figures
├── run_neurips_evals.sh       # One-command reproducibility
└── pyproject.toml             # PyTorch-only dependencies
```

---

## Core Architectures

### 1. PANO (Physics-Anchored Neural Observer)

**File:** `experiments/phase6/hopper_pano.py`

PANO bypasses latent space entirely. During sensor dropout, it predicts observation-space velocity from action history and integrates:

```
obs_estimate = frozen_obs + velocity_predicted × dt × steps_since_dropout
```

**Components:**
- `PANOVelocityPredictor(obs, action_history) → velocity`
- Euler integration during dropout windows

**Training:** Supervised on `(obs, action_history) → velocity` from SAC rollouts.

### 2. Standard Latent JEPA (the one that fails)

**File:** `experiments/phase6/hopper_standard_jepa.py`

Standard JEPA with residual dynamics in latent space:

```
z_t = encoder(obs, obs_prev)
z_{t+1} = z_t + predictor(z_t, action)  # Residual dynamics
velocity = decoder(z_t)                   # Physics anchor
```

**Training:**
- Velocity loss: `λ_vel=10.0 × MSE(v_pred, v_true)`
- Prediction loss: `λ_pred=0.1 × MSE(z_pred, z_target_EMA)`
- Stop-gradient on target encoder (EMA updated, τ=0.996)

**Failure mode:** Despite residual dynamics and aggressive velocity loss, the latent prediction error dominates velocity error by 50–100× and compounds exponentially during multi-step rollout.

### 3. F3-JEPA (Bouncing Ball version)

**File:** `experiments/phase5/f3_jepa.py`

1D version of the JEPA architecture on the bouncing ball environment:
- F3 Encoder: `(x, Δx/dt) → z` (physics-normalized input)
- Works well in 1D (100% vs FD's 80% under dropout)
- Scales poorly to high-dimensional systems

### 4. EKF Baseline

**File:** `experiments/phase6/hopper_pano.py` (integrated)

Extended Kalman Filter with constant-velocity model for state estimation under dropout. Serves as a classical baseline.

---

## Experiments

### Phase 5: Bouncing Ball (1D Validation)

**Purpose:** Establish that JEPA-style prediction helps in the simplest case.

| Experiment | File | Key Finding |
|---|---|---|
| F3-JEPA training | `f3_jepa.py` | 100% success vs FD's 80% under dropout |
| OOD generalization | `h2_validation.py` | Sharp transition at x0=1.5 boundary |
| Seed variance | `f3_jepa.py` | ±9.2% std across 10 seeds |

### Phase 6: MuJoCo Scale-Up (Paper Results)

#### Experiment A: PANO vs All Baselines (`hopper_pano.py`)

Evaluates on Hopper-v4 with contact-triggered sensor dropout (5-step window):
- **Oracle** (no dropout) — upper bound
- **Frozen Baseline** (dropout, no estimation) — lower bound
- **EKF Baseline** (constant-velocity Kalman filter)
- **PANO** (learned velocity + Euler integration)

All methods use the same SAC oracle policy (1M training steps).
Results include 100 episodes per method with Welch's t-tests and 95% bootstrap CIs.

#### Experiment B: Bulletproof Negative Protocol (`bulletproof_negative.py`)

Three converging experiments validating the negative result:

1. **Data Scaling Law** — More data (10k→100k transitions) does NOT fix prediction/velocity loss gap. The ratio stays 50–100×. This is architectural.

2. **Multi-Environment Ablation** — Standard Latent JEPA fails on:
   - Hopper-v4 (hybrid contact)
   - Walker2d-v4 (bipedal)
   - HalfCheetah-v4 (contact-rich)
   - InvertedDoublePendulum-v4 (smooth, NO hybrid dynamics)
   Failure on smooth systems proves this is NOT hybrid-specific.

3. **Impact Horizon Profiling** — Latent prediction error grows exponentially with rollout steps, in both air and contact phases.

---

## Statistical Methodology

All results use proper statistical testing via `src/evaluation/stats.py`:

- **Welch's t-test** (unequal variance) for comparing methods
- **Bootstrap confidence intervals** (10,000 resamples, 95%)
- **Cohen's d** effect size for practical significance
- **Multiple comparisons** reported (PANO vs Frozen, PANO vs EKF, EKF vs Frozen)

---

## Critical Implementation Details

### Loss Balancing (CRITICAL)

```python
lambda_vel = 10.0   # HIGH: velocity precision anchors physics
lambda_pred = 0.1   # LOW: don't let latent prediction dominate

# WRONG (degrades performance):
# lambda_vel = 1.0, lambda_pred = 1.0
```

The aggressive loss imbalance is necessary because the prediction loss lives in ungrounded latent space while velocity loss has physical meaning.

### EMA + Stop-Gradient

```python
z_target = target_encoder(x_next, x)
z_target = z_target.detach()  # STOP-GRADIENT
loss = MSE(z_pred, z_target)

# Target encoder EMA update (τ=0.996)
for tp, ep in zip(target.params(), encoder.params()):
    tp.data.mul_(τ).add_(ep.data, alpha=1-τ)
```

Without stop-gradient, the representation collapses.

### Contact-Triggered Dropout

```python
# Detect large state changes (contact proxy)
delta = |obs - obs_prev|.max()
if delta > threshold:
    dropout_countdown = dropout_duration
    frozen_obs = obs
```

The velocity threshold (0.1) is a proxy for contact events, not true contact detection.

---

## Figure Generation

Figures are generated **entirely from JSON results** (no hardcoded numbers):

```bash
# Generate figures from experiment results
python experiments/phase6/neurips_figures.py \
    --results-dir results/phase6 \
    --output-dir results/phase6
```

| Figure | Description | Data Source |
|---|---|---|
| `figure1_latent_drift.pdf` | Exponential divergence of latent rollout | Impact profiling |
| `figure2_data_scaling.pdf` | Data scaling asymptote (50–100× ratio) | Data scaling experiment |
| `figure3_performance_recovery.pdf` | PANO vs baselines with statistical bars | Hopper evaluation |
| `figure4_multi_env.pdf` | Multi-environment ablation | Continuous control ablation |

---

## Reproducing Results

### Prerequisites

```bash
pip install torch gymnasium[mujoco] stable-baselines3 scipy matplotlib seaborn tqdm
```

### One-Command

```bash
./run_neurips_evals.sh --seed 42 --episodes 100 --oracle-steps 1000000
```

### Hardware Used

- GPU: NVIDIA RTX 5080
- Python: 3.10+
- PyTorch: 2.x
- CUDA: 12.x

---

## Known Limitations

1. **Single-policy evaluation:** All Hopper experiments use one SAC oracle. Different policies may have different dropout sensitivity.
2. **PANO constant velocity assumption:** Euler integration breaks down for long dropout windows (>10 steps).
3. **Contact detection is approximate:** The velocity-threshold trigger is a proxy, not ground-truth contact detection.
4. **No Dreamer/TDMPC comparison:** A proper model-based RL baseline would strengthen the paper.
5. **Training variance:** F3-JEPA on bouncing ball still shows ±9.2% std across seeds.

---

## Archived Code

The `archive_phase1_to_4/` directory contains early JAX/Flax experiments and scaffolding that are NOT used in the final paper. These include:
- Original JAX PCP-JEPA model (`pcp_jepa/model.py`)
- O1/O2/O3 loss experiments (`o1_event_consistency.py`, etc.)
- Phase 1–4 experimental scripts

These were archived to avoid confusion between the ambitions of the research plan (event-consistency, horizon-consistency, event-localized uncertainty losses) and the actual empirical results (which focus on the negative finding and PANO baseline).

---

*Last updated: 2026-02-23*
