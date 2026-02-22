# PCP-JEPA Research Program

## North Star Discovery

**Target**: Long-horizon planning fails primarily due to rare but decisive "physics events" (contacts, stick-slip transitions, saturations, topology changes) and miscalibration of uncertainty around them.

A representation learned to be "event-consistent" and "horizon-consistent" enables planners to scale horizons dramatically under perception noise and parameter shifts.

---

## Research Axes

| Axis | Focus | Key Question |
|------|-------|--------------|
| A | Event-Consistent Imagination | Do latents preserve when/where events happen? |
| B | Horizon-Consistent Planning Geometry | Do multi-step rollouts preserve cost gradients? |
| C | Physics-Constrained Uncertainty | Is uncertainty sharp at event boundaries? |
| D | Multi-Scale Latent Dynamics | Do chunk boundaries align with events? |

---

## Phase 0: Discovery Rig (Week 1-2)

### Environments

**Tier 1: Smooth Continuous**
- Pendulum
- Cartpole
- Acrobot
- Spring-mass

**Tier 2: Hybrid/Discontinuous** (CRITICAL)
- Bouncing ball (impact)
- Sliding block with friction (stick-slip)
- Hinge with stiction
- Peg-in-hole simplified (contact)

**Tier 3: Partial Observability + Active Perception**
- Occluded pendulum
- Clutter occlusions
- Camera jitter
- Tasks requiring movement to observe

### Evaluation Harness

**For each task, output:**

1. **Horizon scaling curves**: success/return vs H ∈ {10, 30, 100, 300}
2. **Event timing error**: how often model predicts event too early/late
3. **Event-conditioned calibration**: ECE/NLL for event vs non-event segments
4. **Parameter OOD curves**: performance vs shift magnitude (mass, friction, restitution)
5. **Planner robustness curves**: performance vs observation noise + delay

---

## Phase 1: Establish Failure Law (Week 3-6)

### Experiment 1.1: Prediction vs Planning Decoupling

**Question**: Does better prediction yield better long-horizon planning?

**Method**:
1. Train spectrum of models with varying 1-step MSE quality
2. Run MPC (CEM/MPPI) on each for long horizons
3. Measure correlation between 1-step MSE and success at H=100/300

**Discovery criterion**: If correlation collapses at long horizons → planning needs different representation than prediction.

---

### Experiment 1.2: Event Dominance Test

**Question**: Are failures concentrated around events?

**Method**:
1. Label event times from simulator (contact, slip, impact, saturation)
2. Compute where planned trajectory diverges: event vs non-event

**Measure**:
- Fraction of catastrophic failures preceded by event timing error
- KL divergence in latent dynamics around events vs elsewhere

**Discovery criterion**: If >70% failures are event-linked → clean research wedge.

---

### Experiment 1.3: Parameter Shift × Events

**Question**: Does parameter shift hurt mostly because events shift?

**Method**:
1. Sweep friction/restitution/mass
2. Compare event timing drift vs overall MSE drift

**Discovery criterion**: If OOD failures correlate with event timing drift more than MSE → strong evidence for new objective.

---

## Phase 2: Invent Objectives (Week 7-14)

Test 3 candidate objectives; don't know which will win.

### Objective O1: Event-Consistency Loss

Learn event belief (soft token), enforce:
- Event probability matches simulator event labels (weak supervision) OR
- Event is self-supervised via change-point consistency across views

**Loss components**:
- Event timing alignment (soft DTW or cross-entropy)
- Event-conditioned rollout consistency

**Ablation**: Remove event terms → see horizon collapse

**Potential discovery**: Event-consistent latents dramatically extend planning horizon.

---

### Objective O2: Horizon-Consistency Loss

Train latent so planning gradients are stable.

**Idea**: Actions selected by planner in latent space should remain near-optimal when executed.

**Implementation** (no oracle needed):
1. Plan with model (MPC) → execute in env → compute realized return
2. Update representation to reduce regret for those trajectories

**Metrics**:
- Planning performance improvement per environment step
- Horizon scaling improvements without improving 1-step prediction much

**Potential discovery**: Latents can be "planning-sufficient" even if not predictive in pixel space.

---

### Objective O3: Event-Localized Uncertainty

Force uncertainty to be sharp at event boundaries:
- Calibrated NLL around events
- Low uncertainty away from events

Planner optimizes expected cost + risk penalty only where uncertainty spikes.

**Potential discovery**: Robust long-horizon planning achieved with minimal conservatism if uncertainty is event-localized.

---

## Phase 3: Multi-Scale Imagination (Week 15-20)

### Experiment 3.1: One-step vs Chunked Rollouts

**Compare**:
- Fine-step-only latent dynamics
- Chunk-only dynamics (predict z_{t+K})
- Hybrid multi-scale (fine + chunk consistency)

**Measure**:
- Error growth rate curves
- Horizon planning success

**Discovery criterion**: Multi-scale consistency yields step-change in feasible horizons.

---

### Experiment 3.2: Temporal Abstraction via Events

**Test**: Do chunk boundaries align with events?
- Allow model to choose segmentation
- Check if segmentation correlates with event times

**Discovery criterion**: Event boundaries naturally define right long-horizon abstraction.

---

## Phase 4: Perception (Week 21-28)

### Experiment 4.1: Multi-view JEPA Invariance

Train with multiple camera views + occlusion patterns, enforce:
- View-invariant latent dynamics
- Event consistency across views

**Measure**:
- Video→state probe quality under occlusion
- Planning success under camera jitter

**Discovery criterion**: Event-consistent objectives work when perception is hard.

---

### Experiment 4.2: Active Perception Planning

Task where success requires moving to observe.

Planner optimizes both:
- Task cost
- Information gain proxy (latent uncertainty reduction)

**Measure**: Does planner take "look first" actions without supervision?

**Discovery criterion**: Event-local uncertainty enables emergent information-seeking.

---

## Phase 5: Stress Tests (Week 29-32)

### Stress Suite

- Parameter shifts: friction/restitution/mass/inertia
- Sensor shifts: noise, blur, occlusions, lighting
- Action shifts: delay, saturation, dropped actions
- Environment shifts: new contact surfaces, changed gravity

### Report

- Performance vs shift magnitude curves
- Event timing drift curves
- Calibration curves separated by event/non-event

---

## Phase 6: Raspberry Pi 4B (Week 33-35)

### What to Run

- Encoder + latent dynamics + planner (or distilled planner)

### Measure

- p50/p95/p99 latency
- Energy consumption
- Closed-loop stability under compute jitter

**Potential discovery**: Event-local uncertainty + multi-scale rollout more compute-stable than heavy reconstruction.

---

## Go/No-Go Gates

### Gate G1 (End Phase 1)

**Must demonstrate at least one**:
- Prediction quality decouples from planning at long horizons
- Event timing errors explain most catastrophic failures

**If neither holds**: Pivot to horizon-consistency geometry only.

---

### Gate G2 (Mid Phase 2)

**At least one objective must produce**:
- Visible right-shift in horizon scaling curve (success at H=100 improves)
- Under parameter shift, degradation smoother than baselines

**If none do**: Objective isn't targeting right failure mode.

---

### Gate G3 (Phase 4)

**Gains must survive perception noise/occlusion**, not just state inputs.

---

## Publishable Discoveries (Any One)

1. Event consistency is dominant predictor of long-horizon planning success (new empirical law)
2. Training for event-consistency yields large horizon scaling improvements even when 1-step MSE barely changes
3. Uncertainty must be localized to events to avoid overly conservative planning
4. Multi-scale rollouts effective only when aligned with event boundaries

---

## Baselines (Required)

- Reconstruction world model + MPC (Dreamer-like)
- Plain JEPA embedding prediction + MPC
- VAML (value-aware model learning)
- PLDM (latent planning, reconstruction-free)
- Hybrid-mode baseline for contacts

---

## Immediate Action (7 Days)

### Day 1-2: Implement Tier 2 Environments

- [ ] Bouncing ball (impact events)
- [ ] Sliding block with stick-slip friction
- [ ] Event logging system

### Day 3-4: Build Evaluation Harness

- [ ] Horizon scaling curves (H=10,30,100,300)
- [ ] Event timing drift metric
- [ ] Event-conditioned calibration

### Day 5-6: Run Experiments 1.1-1.3

- [ ] Train reconstruction world model
- [ ] Train JEPA-style predictor
- [ ] Compare on horizon scaling

### Day 7: Produce Failure Law Plots

- [ ] Planning success vs horizon
- [ ] Fraction of failures linked to events
- [ ] Event timing drift vs parameter shift

---

## Deliverables

1. **Benchmark suite**: Event-labeled, horizon-scaling, OOD parameter grids
2. **Metric suite**: Event timing error, event-conditioned calibration, horizon regret curves
3. **Method**: One of O1/O2/O3 emerges as winner
4. **Systems note**: Pi deployment + latency tail behavior

---

## Repository Structure

```
hybrid-world-model/
├── research/
│   ├── RESEARCH_PLAN.md         # This document
│   ├── experiments/
│   │   ├── exp_failure_law.py   # Phase 1 experiments
│   │   ├── exp_objectives.py    # Phase 2 experiments
│   │   └── ...
│   ├── environments/
│   │   ├── bouncing_ball.py     # Tier 2: Impact events
│   │   ├── stick_slip.py        # Tier 2: Friction events
│   │   └── ...
│   ├── evaluation/
│   │   ├── horizon_scaling.py
│   │   ├── event_metrics.py
│   │   └── ...
│   └── results/
│       ├── failure_law/
│       ├── objectives/
│       └── ...
├── src/pcp_jepa/               # Implementation
└── ...
```

---

**Status**: Ready to begin Phase 0 (Discovery Rig construction)

**Next milestone**: Gate G1 - Establish failure law