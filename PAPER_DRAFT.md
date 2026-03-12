# PANO: Physics-Anchored Neural Observer for Contact-Triggered Sensor Dropout

## Abstract
Modern World Models (WMs) and continuous-time reinforcement learning agents struggle when observations are corrupted during contact-rich interactions. Standard methods either rely heavily on prediction (latent divergence) or freeze states (failing due to missing true dynamics). We introduce the Physics-Anchored Neural Observer (PANO), a method that explicitly separates observation regimes into "reliable" (air) and "unreliable" (contact/dropout). PANO leverages a learned forward velocity predictor and Euler integration to accurately estimate states during sensor dropout. Across MuJoCo locomotion environments (Hopper, Walker2d, Ant), we show that PANO recovers near-optimal performance, significantly outperforming sequence WMs (RSSM, TOLD) and recurrent baselines (LSTM). Crucially, an ablation study demonstrates that PANO's inherent smoothing acts as a low-pass filter on contact noise, surprisingly allowing it to outperform even the ground-truth "Oracle" policy in some regimes. Finally, we provide a diagnostic analysis of PANO's failure modes on Walker2d, showing the necessity of scaled capacity and data for highly complex contact dynamics.

## 1. Introduction
High-impact contact events in physical systems (like a robot foot striking the ground) often cause severe transient noise or complete sensor dropout. Current representations in reinforcement learning assume dense, reliable state observations, which leads to exponential error divergence during dropout phases. In this work, we propose PANO.

## 2. Method
### 2.1 The Sensor Dropout Problem
We formulate sensor dropout as a conditionally unobservable Markov Decision Process (MDP). When the true physical contact force $F_c$ exceeds a threshold $\tau$, sensors experience a blackout for $k$ steps.

### 2.2 PANO Architecture
PANO relies on a split-regime approach:
1. **Clear Phase (No Contact):** Observations are trusted. PANO updates its action history buffer.
2. **Dropout Phase (Contact):** PANO utilizes a deep Multi-Layer Perceptron (MLP) trained to predict state velocity $v_t$ from the last known observation $x_{t-k}$ and the action history buffer $H_t$. 
The state is integrated via Euler method: $\hat{x}_t = x_{t-k} + \hat{v}_t \cdot dt \cdot k$.

This physically-grounded inductive bias prevents the exponential latent drift seen in sequence models like RSSM.

## 3. Experiments
### 3.1 Environments and Baselines
We evaluate on MuJoCo Hopper, Walker2d, and Ant-v4 environments under a 5-step contact-triggered dropout regime. 
**Baselines include:**
- **Oracle:** Full access to perfect state.
- **Frozen Baseline:** Zero-order hold (naive freezing of last known state).
- **RSSM & TOLD:** Proxies for state-of-the-art World Models (Dreamer-style and TD-MPC-style).
- **SMA (Simple Moving Average):** A naive filtering baseline.
- **LSTM Observer:** A standard recurrent baseline predicting the current state.
- **EKF:** Extended Kalman Filter.

### 3.2 Main Results
PANO accurately stabilizes the agent. On Hopper and Ant, PANO closely matches or exceeds Oracle performance, dramatically outperforming Frozen and sequence WMs, which suffer from latent drift and catastrophic failure upon contact. The LSTM baseline, while better than Frozen, fails to capture the exact contact physics without the explicit integration bias.

### 3.3 Ablation Study: Why PANO Beats Oracle?
A surprising finding in early experiments was PANO outperforming the Oracle. We hypothesized that raw contact phases inject high-frequency noise that harms the underlying policy. We tested an **Oracle Smoothing Ablation**, where Simple Moving Average (SMA) and Exponential Moving Average (EMA) filters were applied to the Oracle's perfect observations. We found that smoothing the Oracle's observations directly improved its performance, mirroring the inherent filtering effect of PANO's Euler integration during dropout. 

### 3.4 Diagnostic Analysis: Walker2d Failure
While PANO succeeds on Hopper and Ant, it exhibited performance drops on Walker2d. Our diagnostic experiments scaling the MLP depth and the training data volume (from 300 to 1000 episodes) reveal that Walker2d's complex interleaved contact forces (two legs instead of one, multiple concurrent impacts) require exponentially more data and model capacity to reliably predict the velocity vector. 

## 4. Discussion and Limitations
PANO demonstrates the value of explicit, physics-anchored structure over pure sequence modeling for short-horizon transient failures. However, it is limited by its deterministic nature and reliance on a fixed integration step. Walker2d reveals that purely data-driven velocity prediction scales poorly with simultaneous contact points. 

## 5. Future Work
Future iterations of PANO will replace deterministic Euler integration with stochastic differential equations (SDEs) or implicit event-consistent losses to handle multi-contact branching possibilities.

---
*Note: This draft summarizes the findings of the multi-seed experimental suite located in `/results/neurips/`.*
