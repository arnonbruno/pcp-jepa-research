# PCP-JEPA Research: Physics-Informed World Models for Continuous Control

## 1. Overview and Purpose

This repository (**PCP-JEPA**) is an empirical investigation into the failure modes of modern world models (WMs) under contact-triggered sensor dropout in continuous control. It formulates dropout as a conditionally partially observed MDP, demonstrating that standard latent-space world models (like RSSM and TOLD) fail catastrophically when subjected to contact-triggered observation dropout.

As a constructive solution, this project introduces **PANO (Physics-Anchored Neural Observer)**. PANO shifts away from deep latent imagination and instead uses a learned, observation-space velocity predictor combined with explicit Euler integration. PANO achieves a **+160.9% improvement** over a frozen baseline on Hopper-v4, highlighting that short-horizon observation-space estimation is highly effective for contact-rich robotics.

## 2. Project Structure

The codebase is organized into modular components separating architectures, experimental protocols, and results:

```
.
├── experiments/
│   ├── phase5/           # 1D Bouncing Ball diagnostics isolating variance pathology
│   ├── phase6/           # Main Hopper locomotion + multi-env baseline scripts
│   └── bulletproof/      # Full validation protocols and ablations
├── src/
│   ├── models/           # Architectures (PANO, JEPA variants, EKF)
│   ├── evaluation/       # Statistical significance tests
│   └── utils/            # Data processing and training helpers
├── results/
│   ├── phase6/           # Primary single-seed evaluations (100 episodes) & generated figures
│   └── neurips/          # Aggregated multi-seed logs for Hopper, Walker2d, and Ant
├── run_experiments.py    # Main runner for the multi-seed NeurIPS suite
└── run_all_extras.py     # Runner for auxiliary baselines (LSTM, ablation, diagnostics)
```

## 3. Key Components and Their Responsibilities

* **`PANOVelocityPredictor` (`src/models/pano.py`)**: An MLP-based velocity estimator with action history context. It predicts velocity from the last clear observation, bridging the dropout gap via explicit short-horizon integration.
* **`StandardLatentJEPA` (`src/models/jepa.py`)**: Implements a standard residual latent dynamics baseline (`z_next = z + Δz`) utilizing EMA target encoders, serving as a representative proxy for latent-rollout world models.
* **`EventConsistentJEPA` (`src/models/event_jepa.py`)**: An experimental JEPA extension augmented with contact supervision, predicting impulse magnitudes and applying contact-conditioned constraint projections to regularize the latent space.
* **`EKFEstimator` (`src/models/ekf.py`)**: A highly-tunable Extended Kalman Filter for baseline observation-space tracking. It includes routines for auto-calibrating process and measurement noise via trajectory grid-search.

## 4. Technical Patterns and Approaches Used

* **Observation vs. Latent Space**: The project contrasts full latent hallucination with simple, explicit numerical integration (Euler step) in the observation space.
* **Auto-Calibrating Classical Baselines**: The EKF estimator pulls feature scales and heuristic noise levels dynamically from trajectory data before executing an explicit search for Q/R matrices.
* **Conditionally Partially Observed MDPs**: Evaluating agents under rigorous, physics-informed observation disruption (i.e., masking states specifically during high-impact threshold events).
* **Rigorous Statistical Testing**: Evaluation relies heavily on statistical significance (p-values, 95% Confidence Intervals, Cohen's d), emphasizing uncertainty overlap and variance rather than just mean point estimates.

## 5. How to Run the Code

### Prerequisites
Requires Python 3.10+ and standard scientific computing/RL libraries.
```bash
pip install -r requirements.txt
pip install rl_zoo3 huggingface_sb3
```

### Running Experiments
To reproduce the primary 100-episode evaluations on Hopper-v4:
```bash
python experiments/phase6/hopper_pano.py --n-episodes 100
python experiments/phase6/sota_baselines.py --n-episodes 100
```

To run the full multi-seed suite across environments (Hopper, Walker2d, Ant):
```bash
python run_experiments.py
python run_all_extras.py
```

To generate figures from the results:
```bash
python experiments/phase6/neurips_figures.py --results-dir ./results/phase6
python experiments/phase6/generate_main_figure.py
```

## 6. Important Notes About the Implementation

* **Latent Drift:** The central negative result shows that standard latent prediction errors compound exponentially across contact boundaries (-90% performance vs frozen baselines).
* **Overlapping Confidence Intervals:** While PANO's mean return often exceeds the full-observability oracle under dropout, the 95% confidence intervals overlap. This implies the superiority over the oracle is suggestive (possibly acting as a smoothing filter at contact) but not statistically guaranteed.
* **Environment Sensitivity:** Latent rollout is harmful on Hopper-v4 but helps on HalfCheetah-v4. PANO's performance varies on Walker2d-v4, highlighting that velocity prediction with multiple contacts is sensitive to capacity and data scale.
* **Bimodal Failures:** The frozen baseline exhibits extremely high variance due to bimodal outcomes: recovery if dropout occurs during stable phases, and catastrophic collapse during contact phases.