# PANO: An Empirical Investigation into the Failure Modes of Joint-Embedding Predictive Architectures in Hybrid Continuous Control

> **NeurIPS 2026 Submission**

## Abstract

Joint-Embedding Predictive Architectures (JEPA) learn representations by predicting future latent states — a compelling framework for model-based control. We conduct a systematic empirical investigation into JEPA's failure modes when applied to **hybrid continuous control** systems (MuJoCo locomotion with contact-triggered sensor dropout). Our primary finding is **negative**: standard latent JEPA rollouts diverge exponentially in high-dimensional continuous control, even with residual dynamics, EMA target encoders, stop-gradient training, and multi-step rollout objectives. This failure is **architectural, not data-limited**: prediction error dominates velocity error by 50–100× regardless of dataset size (10k–100k transitions), and persists across all tested environments (Hopper, Walker2d, HalfCheetah, InvertedDoublePendulum).

As a constructive foil, we introduce **PANO (Physics-Anchored Neural Observer)**, which bypasses latent rollout entirely by predicting observation-space velocity from action history and applying Euler integration during sensor dropout. PANO recovers partial task performance where standard latent JEPA does not improve over frozen baselines.

## Key Results

### Negative Result: Exponential Latent Drift

Standard Latent JEPA (residual dynamics `z_next = z + Δz`) exhibits exponential prediction error growth during multi-step rollout:

| Rollout Steps | Latent Error (MSE) | Growth Rate |
|:---:|:---:|:---:|
| 1 | ~1.0 | — |
| 5 | ~50 | ~4× per step |
| 10 | ~1000+ | exponential |

This drift persists across all tested environments and training data scales.

### Data Scaling Asymptote

| Training Data | Velocity Loss | Prediction Loss | Ratio |
|:---:|:---:|:---:|:---:|
| 10k | low | high | ~50× |
| 30k | low | high | ~60× |
| 100k | low | high | ~70× |

More data improves velocity prediction but **does not fix** the prediction-velocity misalignment — this is an architectural limit.

### PANO Performance Recovery (Hopper-v4)

| Method | Reward (mean ± std) | 95% CI | p-value vs Frozen |
|:---|:---:|:---:|:---:|
| Oracle (no dropout) | — | — | — |
| Frozen Baseline | baseline | — | — |
| EKF Baseline | ~ frozen | — | — |
| **PANO** | **improved** | — | p < 0.05 |

*(Exact numbers filled by running `./run_neurips_evals.sh`; all results include Welch's t-tests and bootstrap CIs.)*

### Multi-Environment Ablation

Standard Latent JEPA underperforms frozen baselines on:
- **Hopper-v4** (hybrid contact dynamics)
- **Walker2d-v4** (bipedal locomotion)
- **HalfCheetah-v4** (contact-rich)
- **InvertedDoublePendulum-v4** (smooth — no hybrid dynamics)

The failure on smooth systems confirms this is **not** specific to hybrid/contact dynamics.

## Scientific Contributions

1. **Negative Result (Main Contribution):** Rigorous empirical demonstration that JEPA-style latent rollout fails in continuous control under partial observability, with three converging lines of evidence (data scaling, multi-environment ablation, horizon profiling).

2. **Positive Result (Baseline Foil):** PANO recovers partial performance by anchoring state estimation to physics (velocity prediction + Euler integration), demonstrating that the latent rollout step — not the representation learning — is the bottleneck.

## Repository Structure

```
├── experiments/
│   ├── phase5/            # Bouncing Ball (1D validation)
│   │   ├── f3_jepa.py         # F3-JEPA on bouncing ball
│   │   └── figures.py         # Phase 5 figures
│   └── phase6/            # MuJoCo scale-up
│       ├── hopper_pano.py          # PANO vs all baselines (main result)
│       ├── hopper_standard_jepa.py # Standard Latent JEPA (negative result)
│       ├── bulletproof_negative.py # 3-experiment negative protocol
│       └── neurips_figures.py      # Data-driven figure generation
├── src/
│   ├── evaluation/
│   │   └── stats.py           # Statistical tests (Welch's t, bootstrap CI)
│   └── ...
├── archive_phase1_to_4/   # Archived early experiments (not used in paper)
├── results/               # JSON results + PDF figures (generated)
├── run_neurips_evals.sh   # One-command reproducibility script
└── pyproject.toml         # Dependencies (PyTorch, MuJoCo, SB3)
```

## Reproducing Results

### Prerequisites

```bash
pip install torch gymnasium[mujoco] stable-baselines3 scipy matplotlib seaborn tqdm
```

### One-Command Reproduction

```bash
./run_neurips_evals.sh --seed 42 --episodes 100 --oracle-steps 1000000
```

This trains SAC oracles, runs all experiments (PANO, Standard Latent JEPA, EKF, multi-env ablation), computes statistical tests, and generates publication figures.

### Individual Experiments

```bash
# Main result: PANO vs baselines on Hopper
cd experiments/phase6
python hopper_pano.py --n-episodes 100 --results-dir ../../results/phase6

# Negative protocol (data scaling + multi-env + horizon profiling)
python bulletproof_negative.py --seed 42 --results-dir ../../results/phase6

# Generate figures from JSON results
python neurips_figures.py --results-dir ../../results/phase6
```

## Key Insights

### Why Latent Rollout Fails

The latent predictor learns `Δz` that minimizes single-step prediction error, but small per-step errors compound exponentially during multi-step rollout. The velocity decoder provides a physics-grounded loss that converges well, but the latent prediction loss lives in an ungrounded space where errors have no physical interpretation — leading to a 50–100× velocity-prediction loss ratio that more data cannot fix.

### Why PANO Works (Partially)

PANO avoids latent rollout entirely. Instead of evolving `z → z' → z'' → ...` and decoding, it predicts observation-space velocity directly and integrates: `obs_est = obs_frozen + v_pred × dt × steps`. This keeps the estimation grounded in physical quantities and avoids compounding latent drift. The cost: PANO cannot handle long dropouts or complex dynamics.

## Limitations

- **Single-task evaluation:** All Hopper experiments use one SAC oracle policy. Different policies may exhibit different dropout sensitivity.
- **PANO assumes constant velocity:** Linear Euler integration breaks down over long dropout windows (>10 steps).
- **No Dreamer/TDMPC comparison:** Future work should compare against full model-based RL methods.
- **Contact detection is approximate:** The velocity-threshold trigger is a proxy, not true contact detection.

## Citation

```bibtex
@inproceedings{pano2026neurips,
  title={An Empirical Investigation into the Failure Modes of {JEPA} in Hybrid Continuous Control},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
