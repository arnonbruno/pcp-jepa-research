# PANO: Physics-Anchored Neural Observer for Contact-Triggered Sensor Dropout

## Abstract
Modern world models (WMs) and continuous-time reinforcement learning agents struggle when observations are corrupted during contact-rich interactions. Standard approaches either rely heavily on latent prediction (leading to rollout drift) or freeze states (which discards short-horizon dynamics). We introduce the **Physics-Anchored Neural Observer (PANO)**, a method that separates observation regimes into “reliable” (no contact) and “unreliable” (contact / dropout). PANO uses a learned velocity predictor with Euler integration to propagate the last trusted observation during dropout. On **Hopper-v4**, a 100-episode evaluation shows large gains over a frozen observation baseline and over simplified latent WM proxies (RSSM-, TOLD-style), with statistics reported in `results/phase6/`. **Standard latent JEPA rollout** (latent state rolled forward during dropout—not PANO) is **harmful on Hopper** and **inconclusive relative to frozen observations on Walker2d-v4** in the bulletproof ablation (`results/phase6/bulletproof_results.json`); it **helps on HalfCheetah-v4** under the same script. Multi-seed **PANO** and SOTA-proxy runs for **Hopper-v4, Walker2d-v4, and Ant-v4** are logged under `results/neurips/` (`aggregated_summary.json` aggregates Hopper and Walker2d). An **LSTM** observer baseline is implemented in `experiments/phase6/run_new_baselines.py`; **checked-in JSON rewards for LSTM currently exist for Ant-v4 only** (`results/neurips/new_baselines_Ant-v4_seed*.json`). On Hopper, PANO’s **mean** return can exceed the **oracle** (no dropout) in the primary table, but **95% confidence intervals overlap**, so we treat any “above oracle” story as **hypothesis-level**, not established superiority.

## 1. Introduction
High-impact contact events (e.g., foot strike) often cause transient noise or sensor dropout. Dense-observation assumptions break: latent rollouts diverge, and naive freezing loses velocity information. This draft separates **PCP-JEPA** (the research program), **PANO** (observation-space velocity integration), and **latent JEPA rollout** (the latent-state baseline in bulletproof / drift analyses).

## 2. Method
### 2.1 The Sensor Dropout Problem
We formulate dropout as a conditionally partially observed MDP: when contact intensity exceeds a threshold, observations are withheld for \(k\) steps.

### 2.2 PANO Architecture
1. **Clear phase:** Trust observations; maintain a short action history.
2. **Dropout phase:** An MLP predicts velocity from the last clear observation and history; the state is advanced with an explicit Euler step.

This inductive bias targets short dropout horizons without full latent imagination.

## 3. Experiments
### 3.1 Environments and baselines
- **Primary table (Hopper-v4, 100 eps):** Oracle, frozen, EKF, simplified RSSM/TOLD, PANO — `results/phase6/hopper_pano_results.json`, `sota_baselines_results.json`.
- **Multi-env latent JEPA rollout (seed 42):** Hopper, Walker2d, HalfCheetah — `results/phase6/bulletproof_results.json` (column `jepa_reward_mean` vs frozen).
- **Multi-seed PANO / SOTA proxies:** `results/neurips/pano_*_seed*.json`, `sota_baselines_*_seed*.json` (includes **Ant-v4**).
- **LSTM / SMA:** Code in `experiments/phase6/run_new_baselines.py`; **Ant-v4 LSTM (+ SMA) JSONs** in `results/neurips/new_baselines_Ant-v4_seed*.json` (no checked-in Hopper/Walker LSTM result files at this path).

### 3.2 Main results (Hopper)
PANO strongly improves over frozen observations and over simplified RSSM/TOLD under contact-triggered dropout; see README/DOCUMENTATION tables for means, CIs, and tests.

### 3.3 Latent JEPA rollout across environments
From `bulletproof_results.json`: **Hopper** — latent JEPA rollout is significantly below frozen (\(p \approx 0.0065\)). **Walker2d** — mean reward is similar to frozen; **not significant** (\(p \approx 0.37\)) → **neither a clear success nor a Hopper-style failure** under that test. **HalfCheetah** — large gain vs frozen (\(p \ll 0.001\)).

### 3.4 Walker2d and Ant (PANO)
`results/neurips/aggregated_summary.json`: **Walker2d-v4** — PANO mean return is **far below oracle** and **not significantly above frozen** on reward across the aggregated seeds (\(p \approx 0.46\)). Diagnostic runs: `results/neurips/walker_diagnostics_seed*.json`. **Ant-v4** — per-seed PANO and baseline JSONs are in `results/neurips/` (no Ant row in `aggregated_summary.json`; aggregate locally if needed).

### 3.5 Oracle vs PANO (cautious interpretation)
The single-table Hopper comparison shows a higher **mean** for PANO than oracle but **overlapping 95% CIs**. Framing as “low-pass / smoothing” remains speculative until replicated with more seeds or pre-registered tests.

## 4. Discussion and limitations
PANO illustrates that **explicit short-horizon observation-space dynamics** can outperform **latent rollout** in harsh contact settings, but gains are **environment- and protocol-dependent**. Walker2d highlights **capacity / data** sensitivity for velocity prediction with multiple contacts. Deterministic integration is a limitation.

## 5. Future work
Stochastic or event-aware integration, richer contacts, and unified aggregation scripts for all `results/neurips/` environments.

---
*Artifacts: primary Hopper bundle `results/phase6/`; multi-seed / multi-env logs `results/neurips/`.*
