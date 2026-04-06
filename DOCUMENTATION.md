# PCP-JEPA Research Documentation

## Overview

This repository (**PCP-JEPA**) studies physics-informed world models for continuous control under sensor dropout. The main finding is that **latent-space proxies (simplified RSSM/TOLD) and standard latent JEPA rollout fail catastrophically on contact-rich Hopper**, while a simple observation-space velocity predictor (**PANO**, Physics-Anchored Neural Observer) achieves strong performance on the primary Hopper benchmark. **PANO** is the constructive method; **JEPA** in experiment logs refers to **latent JEPA rollout** (latent state rolled forward during dropout), not PANO.

## Experimental Setup

### Environment
- **Hopper-v4** (MuJoCo)
- Contact-triggered observation dropout (5 steps)
- 100 evaluation episodes per method
- Seed: 42 (reproducible)

### Methods Compared
1. **PANO** - Physics-Anchored Neural Observer (observation-space velocity prediction)
2. **Oracle** - SAC expert with no dropout (upper bound)
3. **Frozen Baseline** - SAC expert with frozen observations during dropout
4. **EKF** - Extended Kalman Filter with calibrated Q/R matrices
5. **Simplified TOLD** - TD-MPC2-style latent dynamics
6. **Simplified RSSM** - DreamerV3-style recurrent state space model

---

## Main Results (100 Episodes)

### Performance Summary

| Method | Reward | 95% CI | vs Frozen | Statistical Significance |
|:-------|:-------|:-------|:----------|:-------------------------|
| **PANO** | **1201.1** | [1126, 1282] | **+160.9%** | p=7.9e-33, d=2.07 |
| Oracle | 1116.5 | [1106, 1132] | +142.5% | upper bound |
| Frozen | 460.5 | [401, 522] | baseline | — |
| TOLD | 110.7 | [99, 122] | -76.0% | p=2.2e-19 |
| EKF | 93.3 | [85, 102] | -79.7% | p=1.2e-20 |
| RSSM | 46.3 | [41, 53] | -90.0% | p=7.3e-24 |

### Key Findings

#### 1. PANO vs Oracle (means vs uncertainty)
On the main Hopper table, PANO has a higher **mean** return than the oracle (~+7.6%), but the **95% CIs overlap** (PANO [1126, 1282] vs Oracle [1106, 1132]). Report this as **suggestive**; do not claim strict superiority over full observability without more seeds or narrower intervals. A possible mechanism, if real, is mild smoothing at contact from integrated velocity prediction.

#### 2. Latent-Space World Models Fail
Both RSSM and TOLD perform **worse than doing nothing**:
- RSSM: -90% vs frozen baseline
- TOLD: -76% vs frozen baseline

The latent rollout compounds errors exponentially during contact discontinuities.

#### 3. EKF Cannot Handle Contacts
Even with grid-search calibration over process/measurement noise, the EKF achieves only 93.3 reward (-80% vs frozen). Linear filtering cannot track nonlinear contact dynamics.

#### 4. Frozen Baseline is Bimodal
High variance (σ=310.5) indicates two failure modes:
- Recovery when dropout hits during stable gait
- Catastrophic collapse when dropout hits during contact

---

## Architecture Details

### PANO (Physics-Anchored Neural Observer)

```
Input: observation_t (partial during dropout)
Output: velocity_t (predicted)

Architecture:
- MLP: [obs_dim, 256, 256, obs_dim]
- Training: MSE on velocity targets
- Integration: Euler step during dropout
```

### Simplified RSSM (DreamerV3-style)

```
- Deterministic RNN hidden state
- Stochastic latent state (Gaussian)
- Decoder reconstructs observations
- Loss: reconstruction + KL divergence
```

### Simplified TOLD (TD-MPC2-style)

```
- Joint-embedding predictive architecture
- Latent dynamics model
- Reward prediction head
- Loss: prediction + consistency
```

---

## File Structure

```
results/phase6/
├── hopper_pano_results.json      # Main PANO results (100 episodes)
├── sota_baselines_results.json   # Simplified RSSM/TOLD (100 episodes)
├── bulletproof_results.json      # Multi-env latent JEPA rollout ablation (seed 42)
├── figure_main_results.pdf       # Main comparison figure
├── figure_performance_breakdown.pdf
├── figure1_latent_drift.pdf
├── figure2_data_scaling.pdf
├── figure3_performance_recovery.pdf
└── figure4_multi_env.pdf

results/neurips/
├── aggregated_summary.json       # Multi-seed aggregates (Hopper-v4, Walker2d-v4)
├── pano_*_seed*.json             # PANO per env/seed
├── sota_baselines_*_seed*.json   # RSSM/TOLD/EKF/Oracle/Frozen per env/seed
└── new_baselines_Ant-v4_*.json   # SMA/LSTM (Ant-v4 only, subset of seeds)
```

---

## Reproducibility

### Requirements
```bash
pip install torch gym mujoco-py stable-baselines3 huggingface_sb3
pip install matplotlib seaborn numpy scipy
```

### Run Experiments
```bash
# Main PANO experiment (100 episodes, ~15 min)
python experiments/phase6/hopper_pano.py --n-episodes 100 --seed 42

# SOTA baselines (100 episodes, ~15 min)
python experiments/phase6/sota_baselines.py --n-episodes 100 --seed 42

# Generate figures (use the results dir each script documents in its argparse help)
python experiments/phase6/neurips_figures.py --results-dir ./results/phase6
python experiments/phase6/generate_main_figure.py
```

Multi-seed logs and aggregates live under `./results/neurips/`; that tree is **not** interchangeable with `./results/phase6/` for aggregation.

### Expected Results
With seed=42, you should reproduce:
- PANO: ~1200 reward
- Oracle: ~1116 reward
- Frozen: ~460 reward
- EKF: ~93 reward
- TOLD: ~110 reward
- RSSM: ~46 reward

---

## Citation

```bibtex
@misc{pcp_jepa_2026,
  title={Latent Rollout Under Contact-Triggered Dropout: A Negative Result and a Simple Solution},
  author={Santos, Bruno},
  year={2026},
  note={Preprint; submitted to NeurIPS 2026}
}
```

---

## Changelog

### 2026-03-12
- Ran full 100-episode experiments for all methods
- Updated results with statistical significance tests
- Generated publication-ready figures
- Updated documentation to reflect final results

### 2026-03-11
- Fixed reproducibility issues (seeding, import paths)
- Renamed fake SOTA baselines to "Simplified"
- Removed physics overclaims from EventConsistentJEPA
- Fixed contact detection to use MuJoCo cfrc_ext
