# Archived Phase 1-4 Experiments

This folder contains early JAX/Flax scaffolding, Phase 1-4 experiments, and historical intermediate scripts that are NOT used in the final NeurIPS submission.

## Contents

### JAX Models (Unused)
- `pcp_jepa/` — Original JAX implementation of PCP-JEPA
- `o1_event_consistency.py` — Event consistency loss experiments
- `o2_horizon_consistency.py` — Horizon-based experiments
- `o3_event_uncertainty.py` — Uncertainty quantification attempts

### Phase 1-4 Experiments
- `phase1/` — Initial open-loop planning experiments
- `phase2/` — Early architecture exploration
- `phase3/` — Controller variants
- `phase4/` — Diagnostic experiments

### Supporting Files
- `forensics/` — JAX-based debugging scripts
- `forensics_root/` — Root-level diagnostic data
- `scripts/` — Old automation scripts
- `src_scaffolding/` — JAX-based src/ files (planning_eval.py, loss_schedules.py, etc.)
- `PHASE2_PLAN.md` — Historical Phase 2 planning document

## Why Archived

The final paper focuses exclusively on:

1. **Phase 5**: Bouncing Ball (F3-JEPA 1D validation)
2. **Phase 6**: MuJoCo scale-up (Standard Latent JEPA failure + PANO recovery)

All active code is in `experiments/phase5/` and `experiments/phase6/`.

The original research plan (RESEARCH_PLAN.md) envisioned O1/O2/O3 objectives and
multi-scale dynamics, but the actual empirical results went in a different direction —
focusing on documenting the negative result (JEPA rollout fails) and PANO as a
constructive baseline.

---

*Archived: 2026-02-23*
