# PCP-JEPA Research: Planning-Consistent Physics JEPA

## North Star Discovery

Long-horizon planning fails primarily due to rare but decisive "physics events" (contacts, stick-slip transitions, saturations) and miscalibration of uncertainty around them.

**A representation learned to be "event-consistent" and "horizon-consistent" enables planners to scale horizons dramatically under perception noise and parameter shifts.**

## Research Axes

| Axis | Focus | Key Question |
|------|-------|--------------|
| A | Event-Consistent Imagination | Do latents preserve when/where events happen? |
| B | Horizon-Consistent Planning Geometry | Do multi-step rollouts preserve cost gradients? |
| C | Physics-Constrained Uncertainty | Is uncertainty sharp at event boundaries? |
| D | Multi-Scale Latent Dynamics | Do chunk boundaries align with events? |

## Phases

| Phase | Duration | Goal |
|-------|----------|------|
| 0 | Week 1-2 | Build discovery rig |
| 1 | Week 3-6 | **Establish failure law** |
| 2 | Week 7-14 | Invent objectives (O1, O2, O3) |
| 3 | Week 15-20 | Multi-scale dynamics |
| 4 | Week 21-28 | Perception + partial observability |
| 5 | Week 29-32 | Stress tests |
| 6 | Week 33-35 | Raspberry Pi 4B evaluation |

## Go/No-Go Gates

- **G1** (End Phase 1): Prediction decouples from planning OR event dominance confirmed
- **G2** (Mid Phase 2): Objective produces horizon scaling improvement
- **G3** (Phase 4): Gains survive perception noise

## Project Structure

```
pcp-jepa-research/
├── README.md
├── RESEARCH_PLAN.md           # Full research program
├── src/
│   ├── environments/          # Simulation environments
│   │   ├── tier1_smooth.py    # Pendulum, Cartpole, etc.
│   │   ├── tier2_hybrid.py    # BouncingBall, StickSlip, etc.
│   │   └── tier3_perception.py
│   ├── models/                # World models
│   │   ├── baselines/
│   │   │   ├── reconstruction.py
│   │   │   ├── jepa_baseline.py
│   │   │   └── dreamer_style.py
│   │   └── pcp_jepa/          # Main method
│   │       ├── belief.py
│   │       ├── dynamics.py
│   │       ├── events.py
│   │       ├── planner.py
│   │       └── losses.py
│   ├── evaluation/
│   │   ├── horizon_scaling.py
│   │   ├── event_metrics.py
│   │   └── calibration.py
│   └── utils/
├── experiments/
│   ├── phase1/
│   │   ├── exp1_1_prediction_planning.py
│   │   ├── exp1_2_event_dominance.py
│   │   └── exp1_3_parameter_shift.py
│   ├── phase2/
│   ├── phase3/
│   └── notebooks/
├── results/
├── scripts/
│   ├── run_phase1.sh
│   └── evaluate_all.sh
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
# Install
pip install -e .

# Run Phase 1 experiments
python experiments/phase1/run_all.py

# View results
ls results/phase1/
```

## Key Publications (Target)

1. "Event Consistency is the Dominant Predictor of Long-Horizon Planning Success"
2. "Planning-Consistent Representations Enable Horizon Scaling Without Prediction Improvement"
3. "Event-Localized Uncertainty Avoids Overly Conservative Planning"

## Baselines

- Reconstruction world model + MPC (Dreamer-like)
- Plain JEPA embedding prediction + MPC
- VAML (value-aware model learning)
- PLDM (latent planning, reconstruction-free)
- Hybrid-mode baseline for contacts

## Hardware

- Development: PC with CUDA
- Deployment: Raspberry Pi 4B (real-time evaluation)

## License

MIT