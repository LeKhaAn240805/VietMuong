#!/usr/bin/env bash
# Commands used to reproduce the paper. Run from the repository root.
# Each step can be repeated: finished runs are skipped, unfinished runs resume.
set -euo pipefail
GPUS=${GPUS:-0,1}          # one job per GPU (the paper's setting)
OUT=${OUT:-vimu_runs}

# 0) checks without GPU
python run_vimu_experiments.py --self_test
python -m pytest -q tests/

# 1) plan and estimated cost
python run_vimu_experiments.py --dry_run --gpus "$GPUS" --out_dir "$OUT"

# 2) Table II: 8 models x {sentence-only, word mixing} x 2 directions x seeds 42-44
#    (the copy and rule-based baselines run on CPU in the same command)
python run_vimu_experiments.py --gpus "$GPUS" --out_dir "$OUT" --experiments E1_main

# 3) Table III and Sec. V-C: ViT5-base data strategies, controls, vocabulary expansion
python run_vimu_experiments.py --gpus "$GPUS" --out_dir "$OUT" --experiments E3_strategy

# 4) tables, significance tests and figures
python run_vimu_experiments.py --report_only --out_dir "$OUT"

# A single run, e.g. ViT5-base, Muong -> Vietnamese, sentence-only, seed 42:
#   python run_vimu_experiments.py --gpus 0 --out_dir "$OUT" --experiments E1_main \
#       --only_run_ids vit5-base__m2v__base__80-10-10__s42
