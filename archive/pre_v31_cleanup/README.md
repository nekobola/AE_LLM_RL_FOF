# Pre-V3.1 Cleanup Archive

This directory contains code and result files from earlier iterations of the
EventTrack design, archived on 2026-06-07 when the codebase was streamlined to
focus on **V3** (matrix-normalized exponential tilting) and **V3.1** (V3
audit fixes: matrix scores + AE shifter + b-as-policy).

## What was moved here

### Verification scripts (root-level, not pytest)
- `verify_stage2.py` — Stage 2 regime-conditional verification (the precursor
  to V2's signal-tilted RB). Used to validate the Stage 2 track divergence
  fix. Superseded by V2 → V3 → V3.1 progression.
- `verify_v2_vs_v1.py` — V1 (three-prototype softmax) vs V2 (signal-tilted RB)
  on the same 156 weeks. Result captured in `v2_vs_v1_verification.csv`.
- `verify_v3_vs_v2_vs_v1.py` — 3-way V1/V2/V3 comparison. Result captured in
  `v3_vs_v2_vs_v1_verification.csv`. **Superseded by the 4-way verification
  (`verify_v3_1_vs_v3_vs_v2_vs_v1.py` in the repo root) which adds V3.1.**

### Unit tests (pytest)
- `test_event_track_v2.py` — V2 unit tests (signal-tilted RB behavior). The V2
  implementation (`src/compute/event_track_v2.py`) remains in place as it is
  still imported by `dual_track_engine.py` and used as a comparison baseline
  in the 4-way verification, but its dedicated test suite is archived.
- `test_normal_track_regime.py` — Stage 2 NormalTrack bear-regime tests. Stage
  2 was a regime-conditional refactor of NormalTrack; the current NormalTrack
  already incorporates its key ideas (bear forced reallocation, widened bounds).
- `test_track_divergence.py` — Stage 2 dual-track divergence tests. The
  underlying behavior is now covered by V3/V3.1 unit tests via the
  `DualTrackEngine` integration.

### Result CSVs
- `stage2_verification.csv` — Stage 2 verification raw output (156 weeks).
- `v2_vs_v1_verification.csv` — V1/V2 comparison raw output.
- `v3_vs_v2_vs_v1_verification.csv` — V1/V2/V3 3-way raw output. The 4-way
  raw output is `results/v3_1_vs_v3_vs_v2_vs_v1_verification.csv` (kept).

## What was deliberately NOT moved

- **Source code in `src/compute/`**: all four implementations
  (`event_track.py` = V1, `event_track_v2.py` = V2, `event_track_v3.py` = V3,
  `event_track_v3_1.py` = V3.1) remain in place because
  `dual_track_engine.py` imports all of them and the 4-way verification
  loads all four engines as comparison baselines.
- **V3 / V3.1 unit tests** in `tests/test_event_track_v3.py` and
  `tests/test_event_track_v3_1.py`: these are the core audit tests and must
  remain runnable via `pytest`.
- **4-way verification** `verify_v3_1_vs_v3_vs_v2_vs_v1.py`: this is the
  current audit tool and its CSV is the most recent comparison data.
- **Pre-existing test files** `test_mdp_environment_alignment.py`,
  `test_normalizer_no_lookahead.py`, `test_regime_autoencoder.py`: these test
  unrelated infrastructure (PPO environment, normalizer, regime autoencoder)
  and have no V3/V3.1 dependency.

## Why archive instead of delete

The archived files preserve the design evolution history (V1 prototypes → V2
RB → V3 exp-tilting → V3.1 audit fix) and the empirical results that justified
each transition. Researchers wanting to reproduce the earlier comparisons can
reconstruct the data from these CSVs. Anyone wanting to revert to an earlier
design can copy the relevant files back.

The V3.1 design supersedes V1/V2/V3 on the 3 audit dimensions (scale
mismatch, gold tragedy, AE gain paradox); the older versions are kept only as
comparison baselines.
