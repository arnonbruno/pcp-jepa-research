# Phase 2: Objectives that Kill Event Failures

## Overview

With 97.1% of failures event-linked, we need objectives that target event handling under planning, not just prediction accuracy.

## Objectives

| Objective | Target | Mechanism |
|-----------|--------|-----------|
| O1 | Event-Consistency | Event tokens + timing loss |
| O2 | Horizon-Consistency | Planning regret loss |
| O3 | Event-Localized Uncertainty | Sharp uncertainty at events |

## Event Definition

Per-step event signal y_t ∈ {0,1}:
- **Impact**: contact impulse > threshold OR contact count changes
- **Stick-slip**: sign change in tangential velocity + friction regime switch
- **Saturation**: action clipping active

Event window label (for stability):
- ŷ_t = 1 if any y_{t-Δ:t+Δ} = 1 with Δ = 5

## Base Model Backbone (Constant Across O1-O3)

```
Encoder → z_t
Dynamics: z_{t+1} = f(z_t, a_t, e_t)  # e_t = event token
Planner: MPPI or CEM (fix one for primary)
```

## Evaluation Metrics (Track Always)

1. Event timing error: E[|t_pred - t_true|]
2. Event recall @Δ: detect event within ±Δ steps
3. Catastrophic failure rate
4. Event-linked failure rate
5. Horizon scaling curve: success vs H ∈ {10, 50, 200, 300}
6. Regret curve: realized return vs baseline

---

## O1: Event-Consistency Objective

### Model Additions

- Event head: p_t = σ(h(z_t))
- Event-conditioned dynamics: f(z_t, a_t, p_t)

### Losses

1. **Event classification**: L_evt = BCE(p_t, ŷ_t)
2. **Event timing**: Soft-DTW on event probabilities vs labels
3. **Event-consistent rollouts**: L_seq = Σ BCE(p_{t+i}^{rollout}, ŷ_{t+i})

### Ablations

- O1-a: no event head, no conditioning
- O1-b: event head but not fed into dynamics
- O1-c: event conditioning but no timing loss
- O1-d: exact y_t vs window ŷ_t

### Success Criteria

- Event recall @Δ increases
- Event timing error drops
- Catastrophic failures drop disproportionately
- Horizon success curve shifts right

---

## O3: Event-Localized Uncertainty

### Model Additions

- Gaussian head: μ(z), Σ(z)
- Risk-aware planner: minimize E[cost] + β * risk

### Losses

1. **NLL prediction**: L_nll = -log p(z_{t+1} | z_t, a_t)
2. **Variance shaping**: 
   ```
   L_varshape = E[ŷ_t * ReLU(σ_min_evt - σ_t) + (1-ŷ_t) * ReLU(σ_t - σ_max_non)]
   ```

### Ablations

- O3-a: no varshape (plain NLL)
- O3-b: varshape but risk-neutral planner
- O3-c: risk-aware planner but no varshape

### Success Criteria

- Calibration improves on event windows
- Planner uses uncertainty to avoid brittle timing
- Failure rate drops without global conservatism

---

## O2: Horizon-Consistency Objective

### Mechanism (No Oracle Actions)

1. Plan: a* = MPC(z_t)
2. Execute in simulator → J_true(a*)
3. Compare to model → J_model(a*)
4. Penalize mismatch

### Losses

1. **Cost mismatch**: L_cost = (J_true - J_model)^2
2. **Event-window weighted**: reweight near events by w > 1
3. **Gradient alignment** (optional): ∂J_model/∂a alignment

### Ablations

- O2-a: no event-window weighting
- O2-b: no cost head (only dynamics)
- O2-c: random actions (tests planning signal)

### Success Criteria

- Planning improves even if prediction MSE doesn't
- Horizon scaling improves on event-heavy tasks

---

## Experimental Matrix

### Environments (Minimum 3)

1. **Impact-heavy**: BouncingBall / Hopper
2. **Friction regime**: StickSlipBlock
3. **Manipulator**: Pushing/Peg-in-hole (simplified)

### Runs Per Environment

- B0: Baseline JEPA (no event/uncertainty/planning-consistency)
- O1 full + ablations
- O3 full + ablations
- O2 full + ablations (1-2 envs first)

Seeds: 3 initially, scale on signal

---

## Gate G2 Criteria

Pass if ANY objective achieves ALL:

1. ≥30% reduction in catastrophic failures
2. Meaningful horizon right-shift (H=200 improves)
3. Event-linked failure rate drops
4. Improvements persist under parameter shift OR observation noise

---

## Common Pitfalls

- Event head collapses → class balancing + window labels
- Planner gradients explode → start with cost mismatch only
- Uncertainty becomes globally high → varshape clamps essential
- Tokens help prediction not planning → use long horizons + event-heavy tasks

---

## Execution Order

1. Implement O1 (fastest signal) on 97.1% environment
2. Add O3 (synergizes with O1)
3. Attempt O2 (needs stable harness)