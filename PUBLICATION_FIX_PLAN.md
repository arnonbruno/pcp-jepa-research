# Publication Fix Plan

## Recommended Publication Angle

**Primary angle:** negative-results paper with a constructive baseline.

Why this is the strongest angle today:
- The checked-in artifacts already support a reproducible negative finding: standard latent JEPA is brittle under contact-triggered dropout and exhibits severe multistep drift.
- `PANO` has evidence as a constructive observation-space baseline, but only for a modest Hopper gain (`+83.1%`, `p = 0.568`) rather than a frontier method breakthrough.
- `EventConsistentJEPA` exists as research code but is not yet integrated into Phase 6 results.
- `experiments/phase6/sota_baselines.py` contains simplified stress tests, not official DreamerV3 or TD-MPC2 baselines.

**Do not submit as a full method paper yet** unless the Phase 6 pipeline is re-run with:
- a validated EventConsistentJEPA main table,
- real DreamerV3 and TD-MPC2 integrations,
- contact-grounded ablations with updated statistical results.

## Canonical Narrative

Use one repository-wide story:
- Standard latent JEPA suffers severe multistep drift under dropout in MuJoCo control.
- The effect is clearly harmful on Hopper, inconclusive on Walker2d, and better on HalfCheetah, so the current evidence is environment dependent rather than a strict hybrid-only law.
- PANO is a constructive baseline that improves Hopper performance over frozen observations, but remains far below the oracle.
- DreamerV3 and TD-MPC2 claims must be removed unless official implementations are integrated.
- EventConsistentJEPA must be marked experimental until Phase 6 results exist.

## File-Level Fix Plan

### Already implemented in this pass

1. `src/envs/contact_dropout.py`
- Prefer true MuJoCo contact semantics via `data.contact` and `mujoco.mj_contactForce`.
- Expose richer metadata: `contact_pairs`, `contact_distance_min`, `contact_normal_force_max`, `contact_pair_count`.
- Keep `cfrc_ext` fallback for environments or tests where full MuJoCo APIs are unavailable.

2. `src/models/event_jepa.py`
- Upgrade EventConsistentJEPA from a generic latent regularizer to a contact-supervised variant.
- Add event logits, non-negative impulse prediction, contact-conditioned latent correction.
- Add grounded losses:
  - event supervision loss,
  - contact impulse loss,
  - complementarity loss using contact distance,
  - aggregate physics constraint loss.

3. `src/utils/data.py`
- Add `generate_event_jepa_data(...)` to collect transition data with contact labels, force targets, impulse targets, and contact distance.

4. `src/utils/training.py`
- Add `train_event_consistent_jepa(...)` using the new contact-aware losses.

5. `README.md`
- Rewrite top-level paper framing to a negative-results angle.
- Remove unsupported `+215.4%` method claim.
- Align key Hopper result to the checked-in final report (`+83.1%`).
- Mark DreamerV3 / TD-MPC2 comparisons as unofficial until real integrations exist.

6. `DOCUMENTATION.md`
- Align narrative with the negative-results angle.
- Clarify that EventConsistentJEPA is experimental.
- Update contact-detection description to MuJoCo-grounded semantics.

7. `EXPERIMENT_RESULTS.md`
- Mark it as the canonical repository narrative.

8. `experiments/phase6/bulletproof_negative.py`
- Remove over-claims about proving or disproving hybrid specificity.
- Reframe the experiment as measuring environment dependence.

9. `tests/test_envs.py`
- Add MuJoCo-contact mocking coverage for the new contact metadata path.

10. `tests/test_event_jepa.py`
- Add coverage for logits, impulse prediction, and physics losses.

11. `tests/test_integration.py`
- Add a smoke test for `train_event_consistent_jepa(...)`.

## High-Priority Remaining Work

### A. Integrate or explicitly quarantine EventConsistentJEPA

**Preferred for the current paper:** quarantine from main claims.

Required file changes:
- `README.md`: keep EventConsistentJEPA out of contributions.
- `DOCUMENTATION.md`: keep it marked experimental.
- Optionally add a dedicated Phase 6 script:
  - `experiments/phase6/hopper_event_consistent_jepa.py`
  - Purpose: train/evaluate EventConsistentJEPA using `generate_event_jepa_data(...)` and compare against frozen baseline + standard JEPA + PANO.

Decision gate:
- If new runs show statistically strong gains over standard JEPA and PANO, reconsider method-paper framing.
- If not, leave it as future work and do not headline it.

### B. Replace simplified SOTA stand-ins with real baselines

Required work:
- Replace `experiments/phase6/sota_baselines.py` or split it into:
  - `experiments/phase6/sota_standins.py` for internal stress tests.
  - `experiments/phase6/official_baselines/` for actual integrations.
- Integrate official codepaths or checkpoints for:
  - DreamerV3,
  - TD-MPC2.
- Re-run Hopper dropout evaluation with the same oracle and dropout wrapper.

Required document updates after real runs:
- `README.md`
- `DOCUMENTATION.md`
- `EXPERIMENT_RESULTS.md`
- `results/phase6/sota_baselines_results.json`

### C. Re-run all Phase 6 results with the repaired contact wrapper

Scripts to re-run:
- `experiments/phase6/hopper_pano.py`
- `experiments/phase6/bulletproof_negative.py`
- any future `hopper_event_consistent_jepa.py`

Artifacts to regenerate:
- `results/phase6/hopper_pano_results.json`
- `results/phase6/bulletproof_results.json`
- `results/phase6/sota_baselines_results.json` if SOTA baselines remain in scope
- figures from `experiments/phase6/neurips_figures.py`

### D. Tighten impact profiling

Current risk:
- the checked-in impact profiling results look closer to general multistep drift than a validated air-vs-contact decomposition.

Required file changes:
- `experiments/phase6/bulletproof_negative.py`
  - replace heuristic phase labeling with MuJoCo contact labels from the contact wrapper.
- Optionally factor out a reusable utility:
  - `src/utils/contact_labels.py`

## Submission Readiness Criteria

Before calling this frontier-grade and paper-ready, require all of:
1. One canonical result table regenerated from current code.
2. One canonical paper story consistent across `README.md`, `DOCUMENTATION.md`, `EXPERIMENT_RESULTS.md`, and figure captions.
3. Contact-trigger semantics derived from MuJoCo, not observation deltas.
4. Either:
   - EventConsistentJEPA fully integrated and validated, or
   - EventConsistentJEPA clearly excluded from the main claimed contribution.
5. Either:
   - official DreamerV3 / TD-MPC2 baselines, or
   - no official-comparison claim anywhere in the repo.
