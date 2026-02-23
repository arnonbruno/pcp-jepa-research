# Archived Phase 1-4 Experiments

This folder contains early JAX/Flax scaffolding and Phase 1-4 experiments that were not used in the final NeurIPS submission.

## Contents

### JAX Models (Unused)
- `pcp_jepa/` - Original JAX implementation of PCP-JEPA
- `o1_event_consistency.py` - Event consistency loss experiments
- `o2_horizon_consistency.py` - Horizon-based experiments
- `o3_event_uncertainty.py` - Uncertainty quantification attempts

### Phase 1-4 Experiments
- `phase1/` - Initial open-loop planning experiments
- `phase2/` - Early architecture exploration
- `phase3/` - Controller variants
- `phase4/` - Diagnostic experiments
- `forensics/` - Debugging and analysis scripts
- `notebooks/` - Jupyter notebooks (empty)

## Why Archived

Reviewer feedback indicated these JAX scaffolding files distracted from the core PyTorch empirical results. The final paper focuses exclusively on:

1. **Phase 5**: Bouncing Ball (PANO vs FD)
2. **Phase 6**: Hopper scale-up (Standard Latent JEPA explosion, PANO recovery)

All active code is in `experiments/phase5/` and `experiments/phase6/`.

---

*Archived: 2026-02-23*
