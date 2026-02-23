# Experiment Registry: PCP-JEPA Phase 5

## Hypotheses

### Primary Hypothesis (H1)
In hybrid/event-driven dynamics, the main barrier to long-horizon control from perception is not prediction accuracy, but recovering a control-sufficient belief about event phase and post-event state. A physics/event-informed JEPA can learn a latent belief z_t where a tiny feedback law becomes optimal.

### Secondary Hypotheses
- H2: Event-phase supervision (L2) improves latent's support of simple feedback
- H3: Impact-consistency constraints (L3) improve generalization across restitution values
- H4: Control-sufficiency loss (L4) directly enables linear feedback in latent

## Success Criteria

### Primary Metrics
- Success rate gap closed: (JEPA + linear feedback - open-loop) / (state-PD - open-loop)
- Target: ≥ 60% gap closed in partial-obs regimes (O1, O2)

### Secondary Metrics
- Event-phase prediction accuracy (contact flag, time-to-impact)
- Generalization: success rate on held-out restitution values
- Robustness: success rate under noise/occlusion/dropout

## Datasets

### Environments
- E1: BouncingBall with restitution e ∈ [0.3, 0.95]
- E2: StickSlip with friction knob

### Observation Regimes
- O0: Full state [x, y, vx, vy] (upper bound)
- O1: Partial state - position only + noise/dropout
- O2: Pixels - 64x64 grayscale render

### Data Splits
- Train: seeds 0-499
- Val: seeds 500-749
- Test: seeds 750-999
- Generalization: train on e ∈ {0.3, 0.5, 0.7, 0.85}, test on e ∈ {0.4, 0.6, 0.8, 0.9, 0.95}

## Stopping Criteria

- If H1 fails (JEPA doesn't close >30% gap by Week 3): pivot to stronger L3/L4
- If H2/H3/H4 ablations show no improvement: simplify architecture
- Maximum runtime: 6 weeks from start

## Ablation List

1. Base: JEPA (L1 only)
2. +Event (L1 + L2)
3. +Impact (L1 + L2 + L3)
4. +Control (L1 + L2 + L3 + L4)
5. -Recurrence (per-frame encoding only)
6. -Multistep (k=1 only)

## Run Order

1. Phase 1: Build observation regimes + dataset
2. Phase 2: Baselines (perception gap)
3. Phase 3: JEPA training
4. Phase 4: Full evaluation + ablations

Last updated: 2026-02-23
