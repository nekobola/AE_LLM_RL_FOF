# Restore from pre-theta-refactor backup

Snapshot taken: **2026-06-07 17:53:07**

## When to use

Use this backup to roll back to the pre-Stage 7 design where:
- PPO outputs (a1, a2) → (delta_alpha, delta_tau) for **fusion ratio** between NormalTrack and EventTrack V3.1
- NormalTrack + dual_track_engine are active modules
- V3.1 THETA is a class static (`THETA = 0.7`)
- `metrics_real.json` is the source of truth for real-OHLC performance
- Stage 6 result: V3.1 Sharpe 1.096, V3 Sharpe 0.461 (real OHLC, 5y WFO)

## Restore

### Full restore (overwrite current files)
```bash
cd archive/pre_theta_refactor_20260607
rsync -av --delete \
    src/compute/  ../../src/compute/ \
    src/env/      ../../src/env/ \
    scripts/      ../../scripts/ \
    tests/        ../../tests/ \
    checkpoints/  ../../checkpoints/ \
    docs/         ../../docs/
cp config.yaml        ../../config.yaml
cp verify_stage*.py   ../../
cp results/           ../../results/ -r
```

### Verify restored checksum
```bash
cd archive/pre_theta_refactor_20260607
sha256sum -c MANIFEST.sha256
```

### Verify the pipeline still works
```bash
cd ../..  # project root
python -m pytest tests/test_event_track_v3_1.py tests/test_event_track_v3.py tests/test_ml_stage1_fix.py -v
python verify_stage6.py
```

Expected: V3.1 Sharpe 1.096, V3 Sharpe 0.461 (matches §七 in docs/strategy_details.md).
