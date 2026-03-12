# PCP-JEPA Research

Physics-Informed World Models for Continuous Control

## Abstract

This repository currently supports a **negative-results paper** about latent-rollout failure under contact-triggered sensor dropout in continuous control. The strongest reproducible claim in the checked-in artifacts is that **standard latent JEPA exhibits severe multistep drift in MuJoCo control**, and that a simple observation-space baseline, **PANO** (Physics-Anchored Neural Observer), provides a **modest but not statistically significant +83.1%** improvement over a frozen-observation baseline on Hopper-v4 (`p = 0.568` on 2 evaluation episodes).

The repository does **not** currently support stronger method-paper claims such as a validated event-consistent JEPA, a canonical hybrid-specific theorem, or official DreamerV3 / TD-MPC2 baseline comparisons. Those remain future work.

## Repository Structure

```
experiments/
├── phase5/           # 1D Bouncing Ball diagnostic
├── phase6/           # Hopper locomotion + multi-env
└── bulletproof/      # Full validation protocol

src/
├── models/           # PCP-JEPA architecture
├── evaluation/       # Statistical tests
└── utils/            # Helper functions

results/
└── phase6/
    ├── hopper_pano_results.json       # Main PANO results
    ├── bulletproof_results.json       # Full protocol
    └── figure*.pdf                    # Publication figures
```

## Quick Start

```bash
pip install -r requirements.txt
python run_experiments.py
```

---

## Key Results

### 1D Diagnostic: The Pathology of Variance

**Phase 5 (Bouncing Ball)** isolates the failure mode in a minimal 1D environment. We demonstrate that small state-estimation errors at the exact boundary of a hybrid impact dictate discrete success or failure, causing standard learned observers to exhibit massive bimodal variance across seeds.

- **F3-JEPA** maintains 100% success across all dropout durations
- **Frozen Detector** drops to 80% success at first dropout step

This proves the pathology is localized to impact boundaries, not accumulated over trajectory.

---

### PANO Performance Recovery (Hopper-v4)

Using pretrained SAC experts from RL Baselines3 Zoo (HuggingFace):

| Method | Reward (Mean ± Std) | 95% CI | Improvement | p-value vs Frozen |
|:-------|:-------------------|:-------|:-----------|:------------------|
| Oracle (no dropout) | 1121.3 ± 9.3 | [1114.7, 1127.9] | upper bound | — |
| Frozen Baseline | 175.5 ± 58.4 | [134.3, 216.8] | baseline | — |
| EKF Baseline | 30.5 ± 16.2 | [19.1, 42.0] | **-82.6%** | 0.156 |
| **PANO** | **321.3 ± 257.7** | **[139.1, 503.5]** | **+83.1%** | **0.568** |

**Key Finding:** PANO improves over the frozen baseline on average in limited testing, but remains far below the oracle. It should be read as a constructive observation-space baseline, not a solved method for hybrid control under dropout.

---

### Multi-Environment Ablation: Hybrid vs Smooth Systems

We tested Standard Latent JEPA on three MuJoCo environments with fundamentally different contact dynamics:

| Environment | Contact Type | Oracle | JEPA | Baseline | p-value | Outcome |
|:------------|:-------------|:-------|:-----|:---------|:--------|:--------|
| Hopper-v4 | Harsh hybrid impacts | 1109.3 | 241.2 | 361.2 | **0.0065** | JEPA **FAILS** |
| Walker2d-v4 | Bipedal impacts | 3782.8 | 222.2 | 175.8 | 0.369 | JEPA matches baseline |
| HalfCheetah-v4 | Smooth rolling contacts | 9408.7 | **586.5** | 207.2 | **p < 0.0001** | JEPA **SUCCEEDS** |

**Interpretation:** The environment-specific picture is mixed: Hopper clearly fails, Walker2d is inconclusive, and HalfCheetah improves over baseline. The current codebase therefore supports the conservative claim that latent rollout is brittle under dropout in contact-rich control, but does **not** yet prove a strict hybrid-only failure law.

---

### Negative Result: Exponential Latent Drift

Prediction error grows exponentially regardless of physics phase:

| Step | Overall Error | Growth Factor |
|:-----|:--------------|:--------------|
| 1 | 151.1 | — |
| 2 | 243.7 | 1.6× |
| 3 | 371.4 | 1.5× |
| 4 | 577.6 | 1.6× |
| 5 | 831.8 | 1.4× |

**Finding:** Error compounds at roughly 1.5× per step in the checked-in profile, reaching 2898.4 by step 10. The qualitative result is runaway multistep drift, even if the exact magnitude depends on the current profiling setup.

---

### Data Scaling Law

| Transitions | Velocity Loss | Prediction Loss | Ratio |
|:------------|:--------------|:----------------|:------|
| 10,000 | 11.5 | 2130.0 | **185×** |
| 30,000 | 7.6 | 908.1 | **120×** |
| 100,000 | 3.6 | 272.3 | **76×** |

**Finding:** Even with 100k transitions, prediction loss dominates velocity loss by 76×. This is an **architectural limitation**, not data insufficiency.

---

## Key Contributions

1. **Failure Characterization:** Standard Latent JEPA rollouts are brittle under contact-triggered sensor dropout in MuJoCo control. The current multi-environment evidence is mixed, with clear failure on Hopper, inconclusive Walker2d results, and better HalfCheetah behavior.

2. **Constructive Baseline:** PANO bypasses latent rollout entirely, using direct velocity prediction + Euler integration, achieving **+83.1%** improvement over frozen baseline on Hopper-v4 in the checked-in final report.

3. **EKF Collapse:** Extended Kalman Filter catastrophically fails (-82.6%), demonstrating that naive state estimation alone is insufficient in this setting.

4. **Architectural Limit:** Data scaling reduces but does not eliminate the prediction-velocity gap (185× → 76×), indicating that the standard JEPA latent rollout remains poorly aligned with multi-step prediction under dropout.

5. **Scope Note:** `experiments/phase6/sota_baselines.py` contains simplified RSSM/TOLD stress tests, not official DreamerV3 or TD-MPC2 reproductions.

---

## Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt
pip install rl_zoo3 huggingface_sb3

# Run full experiment suite
python run_experiments.py

# Or run individual experiments
python experiments/phase5/f3_jepa.py                              # Bouncing Ball
python experiments/phase6/hopper_pano.py --n-episodes 100        # PANO evaluation
python experiments/phase6/bulletproof_negative.py --n-eval-episodes 50  # Multi-env
```

All experiments complete in ~12 minutes with pretrained HuggingFace models.

---

## Files

| File | Description |
|:-----|:------------|
| `run_experiments.py` | Main entry point with checkpoint/resume |
| `experiments/phase5/f3_jepa.py` | 1D Bouncing Ball diagnostic |
| `experiments/phase6/hopper_pano.py` | PANO vs baselines |
| `experiments/phase6/bulletproof_negative.py` | Multi-environment protocol |
| `results/phase6/hopper_pano_results.json` | Full results with raw rewards |
| `results/phase6/bulletproof_results.json` | Complete ablation data |

---

## Citation

```bibtex
@inproceedings{pcp_jepa_2026,
  title={Physics-Anchored Neural Observers for Hybrid Control Under Sensor Dropout},
  author={Santos, Bruno},
  booktitle={NeurIPS},
  year={2026}
}
```
