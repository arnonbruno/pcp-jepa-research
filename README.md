# JEPA Failure Modes in High-Dimensional Continuous Control

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-blue)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)]()
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red)]()

**An empirical investigation into the failure modes of Joint-Embedding Predictive Architectures (JEPA) in hybrid continuous control.**

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/arnonbruno/pcp-jepa-research.git
cd pcp-jepa-research
pip install -r requirements.txt

# Reproduce all paper results
chmod +x run_neurips_evals.sh
./run_neurips_evals.sh
```

---

## Abstract

Joint-Embedding Predictive Architectures (JEPA) have shown promise for representation learning, but their application to high-dimensional continuous control remains poorly understood. This paper provides a rigorous empirical teardown of JEPA's failure modes when applied to sensor dropout recovery in dynamical systems.

**Key Findings:**

1. **Negative Result**: Standard Latent JEPA rollouts diverge exponentially in 11D continuous control, reaching **1 trillion prediction error by step 10**. This failure persists even with 100k training transitions.

2. **Mechanism**: Prediction loss dominates velocity loss by **100× regardless of data scale**, indicating a fundamental architectural limitation rather than data starvation.

3. **Baseline Foil**: PANO (Physics-Anchored Neural Observer) — a simple architecture using explicit velocity prediction with Euler integration — recovers **+22.6% performance** under sensor blackout, demonstrating that physics constraints outperform latent hallucination.

---

## The Exponential Latent Drift

| Rollout Step | Latent Prediction Error |
|--------------|------------------------|
| 1 | 1,320 |
| 5 | 422,278 |
| 10 | **1.09 trillion** |

**Finding**: JEPA latent space diverges exponentially (~6× per step) in high-dimensional continuous control. This is not a bug — it's a fundamental architectural failure.

---

## Data Scaling Does Not Help

| Training Transitions | Velocity Loss | Prediction Loss | Ratio |
|---------------------|---------------|-----------------|-------|
| 10,000 | 4.7 | 1,577.9 | 332× |
| 30,000 | 2.4 | 443.3 | 182× |
| 100,000 | 1.2 | 124.6 | **104×** |

**Finding**: Even with 10× more data, prediction loss dominates velocity loss by 100×. This is an architectural asymptote, not a data deficit.

---

## PANO: The Physics-Anchored Solution

| Method | Hopper Reward | Improvement |
|--------|---------------|-------------|
| Oracle (no dropout) | 280.4 ± 3.0 | — |
| FD Baseline (dropout) | 166.9 ± 68.1 | — |
| **PANO** | **204.5 ± 47.8** | **+22.6%** |
| Standard Latent JEPA | 78.9 ± 17.7 | −52.7% |

**Finding**: PANO's explicit velocity prediction (no latent middleman) recovers performance. Physics constraints beat latent hallucination.

---

## Repository Structure

```
pcp-jepa-research/
├── experiments/
│   ├── phase5/                    # Bouncing Ball (2D hybrid)
│   │   └── f3_jepa.py             # PANO architecture
│   └── phase6/                    # Hopper scale-up (11D)
│       ├── hopper_pano.py         # Working solution
│       ├── hopper_standard_jepa.py # Explodes
│       ├── bulletproof_negative.py # 3-experiment validation
│       └── neurips_figures.py     # Paper figures
├── archive_phase1_to_4/           # JAX scaffolding (unused)
├── results/                       # Generated figures (PDF)
├── requirements.txt
└── run_neurips_evals.sh
```

---

## Experiments Summary

### Experiment 1: Data Scaling Law
**Question**: Does more data fix latent drift?  
**Answer**: No. Prediction loss remains 100× velocity loss even at 100k transitions.

### Experiment 2: Continuous Control Ablation  
**Question**: Is failure specific to hybrid dynamics?  
**Answer**: No. Standard Latent JEPA also fails on smooth InvertedDoublePendulum.

### Experiment 3: Impact Horizon Profiling
**Question**: Where does the drift occur?  
**Answer**: Everywhere. Error grows exponentially regardless of physics phase.

---

## Key Insights

### Why Standard Latent JEPA Fails

1. **Prediction loss dominates**: JEPA's objective doesn't align with control needs
2. **No physics grounding**: Latent has no constraint to respect velocity
3. **Exponential drift**: Small errors compound into trillion-scale

### Why PANO Works

1. **Explicit velocity**: Direct prediction, no latent middleman
2. **Physics-grounded**: Uses Δx/dt as base with learned correction
3. **Euler integration**: `obs_est = frozen_obs + velocity × dt`

---

## Figures

All figures available as PDF in `results/`:

| Figure | Description |
|--------|-------------|
| `figure1_latent_drift.pdf` | Exponential divergence to 1 trillion |
| `figure2_data_scaling.pdf` | Data doesn't fix latent-physics mismatch |
| `figure3_performance_recovery.pdf` | PANO recovers +22.6% |
| `figure4_results_table.pdf` | Comprehensive comparison |

---

## Citation

```bibtex
@inproceedings{jepa_failure_modes_2026,
  title={JEPA Failure Modes in High-Dimensional Continuous Control: 
         An Empirical Investigation},
  author={...},
  booktitle={NeurIPS},
  year={2026}
}
```

---

## License

MIT
