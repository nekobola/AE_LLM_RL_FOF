"""Create a complete snapshot of the current (pre-theta-refactor) design.

Backs up source code, configs, PPO checkpoint, WFO results, verify scripts,
and the current strategy_details.md into archive/pre_theta_refactor_20260607/.

Generates:
  - MANIFEST.txt   — list of all backed-up files with SHA256 + size
  - RESTORE.md     — step-by-step restore instructions
  - GITSUMMARY.txt — `git status` + recent commits at backup time

Run from project root:
    python archive/pre_theta_refactor_20260607/BACKUP_NOW.py
"""
from __future__ import annotations
import hashlib
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = PROJECT_ROOT / "archive/pre_theta_refactor_20260607"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Categories of files to back up.
SRC_FILES = [
    "src/compute/normal_track.py",
    "src/compute/event_track.py",
    "src/compute/event_track_v2.py",
    "src/compute/event_track_v3.py",
    "src/compute/event_track_v3_1.py",
    "src/compute/dual_track_engine.py",
    "src/env/action_mapper.py",
    "src/env/mdp_environment.py",
    "src/env/metrics_utils.py",
    "src/env/regret_engine.py",
    "src/env/reward_function.py",
    "src/env/state_assembler.py",
    "scripts/run_backtest_wfo.py",
    "scripts/train_ppo.py",
    "scripts/train_ae.py",
    "scripts/run_inference_live.py",
    "scripts/backfill_metrics_real.py",
    "tests/test_event_track_prototypes.py",
    "tests/test_event_track_v3.py",
    "tests/test_event_track_v3_1.py",
    "tests/test_ml_stage1_fix.py",
    "tests/test_mdp_environment_alignment.py",
    "tests/test_normalizer_no_lookahead.py",
    "tests/test_regime_autoencoder.py",
    "verify_stage4.py",
    "verify_stage5.py",
    "verify_stage6.py",
    "verify_v3_1_vs_v3_vs_v2_vs_v1.py",
    "config.yaml",
    "docs/strategy_details.md",
    "checkpoints/actor_critic.pth",
]

WFO_RESULT_DIRS = [
    "results/wfo/stage5_v3",
    "results/wfo/stage5_v3_1",
    "results/wfo/stage6_v3",
    "results/wfo/stage6_v3_1",
]

WFO_RESULT_FILES = [
    "results/stage4_validation.csv",
    "results/v3_1_vs_v3_vs_v2_vs_v1_verification.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(rel: str) -> tuple[Path, int, str] | None:
    src = PROJECT_ROOT / rel
    if not src.exists():
        return None
    dst = BACKUP_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst, src.stat().st_size, sha256_file(dst)


def copy_dir(rel: str) -> int:
    src = PROJECT_ROOT / rel
    if not src.exists():
        return 0
    dst = BACKUP_DIR / rel
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    n = sum(1 for _ in dst.rglob("*") if _.is_file())
    return n


def write_manifest(items: list[tuple[str, int, str]]) -> None:
    p = BACKUP_DIR / "MANIFEST.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write(f"# Backup manifest — pre-theta-refactor snapshot\n")
        f.write(f"# Created: {TIMESTAMP}\n")
        f.write(f"# Project: {PROJECT_ROOT}\n")
        f.write(f"# Total entries: {len(items)}\n")
        f.write("#\n")
        f.write("# sha256sum verification:\n")
        f.write("#   cd archive/pre_theta_refactor_20260607\n")
        f.write("#   sha256sum -c MANIFEST.sha256\n")
        f.write("#\n")
        f.write(f"{'PATH':<70s}  {'SIZE':>10s}  SHA256\n")
        f.write("-" * 130 + "\n")
        for rel, size, sha in items:
            f.write(f"{rel:<70s}  {size:>10d}  {sha}\n")
    sha_lines = "\n".join(f"{sha}  {rel}" for rel, _, sha in items)
    (BACKUP_DIR / "MANIFEST.sha256").write_text(sha_lines + "\n", encoding="utf-8")


def write_restore_md() -> None:
    p = BACKUP_DIR / "RESTORE.md"
    p.write_text(f"""# Restore from pre-theta-refactor backup

Snapshot taken: **{TIMESTAMP}**

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
rsync -av --delete \\
    src/compute/  ../../src/compute/ \\
    src/env/      ../../src/env/ \\
    scripts/      ../../scripts/ \\
    tests/        ../../tests/ \\
    checkpoints/  ../../checkpoints/ \\
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
""", encoding="utf-8")


def write_git_summary() -> None:
    p = BACKUP_DIR / "GITSUMMARY.txt"
    out = []
    out.append(f"# Git state at backup time: {TIMESTAMP}\n\n")
    out.append("## Status\n")
    try:
        out.append(subprocess.check_output(
            ["git", "status", "--short"], cwd=PROJECT_ROOT, encoding="utf-8"))
    except Exception as e:
        out.append(f"  (git status failed: {e})\n")
    out.append("\n## Recent commits\n")
    try:
        out.append(subprocess.check_output(
            ["git", "log", "--oneline", "-20"], cwd=PROJECT_ROOT, encoding="utf-8"))
    except Exception as e:
        out.append(f"  (git log failed: {e})\n")
    p.write_text("".join(out), encoding="utf-8")


def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, int, str]] = []

    print("Backing up source/config/test files...")
    for rel in SRC_FILES:
        r = copy_file(rel)
        if r is None:
            print(f"  MISSING: {rel}")
            continue
        items.append((rel, r[1], r[2]))
        print(f"  OK: {rel} ({r[1]} bytes)")

    print("\nBacking up WFO result directories...")
    for d in WFO_RESULT_DIRS:
        n = copy_dir(d)
        if n == 0:
            print(f"  MISSING dir: {d}")
        else:
            print(f"  OK dir: {d} ({n} files)")

    print("\nBacking up WFO result files...")
    for f in WFO_RESULT_FILES:
        r = copy_file(f)
        if r is None:
            print(f"  MISSING: {f}")
            continue
        items.append((f, r[1], r[2]))
        print(f"  OK: {f} ({r[1]} bytes)")

    print("\nWriting manifest + restore guide + git summary...")
    write_manifest(items)
    write_restore_md()
    write_git_summary()

    print(f"\n=== Backup complete ===")
    print(f"Location: {BACKUP_DIR}")
    print(f"Files:    {len(items)}")
    print(f"Manifest: {BACKUP_DIR / 'MANIFEST.txt'}")
    print(f"Restore:  {BACKUP_DIR / 'RESTORE.md'}")


if __name__ == "__main__":
    main()
