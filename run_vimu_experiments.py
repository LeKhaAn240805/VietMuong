#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 VietMuong — experiment runner for Vietnamese <-> Muong machine translation
================================================================================

Code for the paper
    "VietMuong: A Bilingual Corpus and Translation Benchmark for
     Vietnamese–Muong Machine Translation" (RIVF 2026).
Dataset: https://huggingface.co/datasets/LKAusagi/VietMuong

One script implements the whole protocol of the paper: data preparation,
baselines, fine-tuning, evaluation, significance tests and report tables.

EXPERIMENTS (--experiments)
    E0  CPU baselines: copy and rule-based dictionary lookup, every split,
        both directions, seeds 42–46                          -> Table II
    E1  8 pre-trained models x {base, word} x 2 directions x seeds 42–44
        base = sentence-only, word = word mixing (all 15,377 lexical pairs)
                                                              -> Table II
    E3  ViT5-base data-strategy ablation, seeds 42–44:
        word_p25 / word_p50 / word_p75   partial dictionaries -> Table III
        ctrl_shuffle / ctrl_dup          matched controls     -> Table III
        token / token_avg                vocabulary expansion -> Sec. V-C
        (word_token duplicates token_avg and is not reported)
    E4  ViT5-base {base, word} at the 70:10:20 and 60:10:30 splits
        (supplementary; not reported in the paper's tables)
    E5  seed-variance study; skipped automatically with >= 3 seeds

    Table I (corpus statistics), Table IV (surface-overlap bands, Δtone) and
    the paired bootstrap tests are written by the same run / --report_only.

PROTOCOL (Sec. III-D and IV-C)
    Data      exact deduplication; near-duplicates grouped by Jaccard >= 0.8
              over syllable 3-gram shingles of the normalised Muong side,
              merged with union–find (12,800 groups); group-level 80:10:10
              split (10,272 / 1,284 / 1,283); lexical pairs only in TRAIN.
    Models    MODEL_REGISTRY: Adafactor for every model, effective batch 16,
              per-model learning rate and epoch cap, BF16 when supported.
    Selection early stopping (patience 3) on BLEU of a fixed subset of 500
              validation pairs; the best checkpoint is restored.
    Test      beam size 4; sacreBLEU (BLEU, 13a, mixed case), chrF++
              (char order 6, word order 2), TER, tone accuracy, Δtone.
    Tests     paired bootstrap, 1,000 resamples, against the same model's
              sentence-only run (and against the copy baseline).
    Latency   seed 42 only, batch 1, beam 4, first 100 test sentences,
              measured inside the run right after test decoding, one run
              per GPU.

USAGE (see README.md for the full reproduction guide)
    python run_vimu_experiments.py --self_test            # no GPU, no data
    python run_vimu_experiments.py --dry_run --gpus 0,1   # print the plan
    python run_vimu_experiments.py --gpus 0,1             # run everything
    python run_vimu_experiments.py --gpus 0 --experiments E1_main
    python run_vimu_experiments.py --report_only          # tables + plots

    By default the data are downloaded from the Hugging Face Hub
    (LKAusagi/VietMuong, pinned revision); --sentence_file / --word_file use
    local CSV files with the same columns instead.

    Runs resume automatically: a run with results/<run_id>.json is skipped, an
    unfinished run continues from its last checkpoint. Checkpoints are deleted
    after each run unless --keep_checkpoints is given.

    CUDA_VISIBLE_DEVICES and `taskset` set by the caller are respected; the
    number of CPU threads is read from the process affinity.
================================================================================
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# SECTION 0 — Constants & registry
# ==============================================================================

def available_cpus() -> int:
    """Number of CPUs this process may actually use.

    Matters under `taskset -c 0-15`: os.cpu_count() still returns every core
    on the machine, which oversubscribes the allowed cores and slows training
    down. sched_getaffinity() gives the correct number.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def detect_gpus() -> str:
    """Detect the available GPUs, respecting a pre-set CUDA_VISIBLE_DEVICES."""
    try:
        import torch
        n = torch.cuda.device_count()
        if n > 0:
            return ",".join(str(i) for i in range(n))
    except Exception:
        pass
    return "0"


def setup_compute_env(gpu_arg: Optional[str], thread_arg: Optional[int]) -> Dict[str, Any]:
    """Respect CUDA_VISIBLE_DEVICES and taskset set by the caller.

    Example:  CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 python run_vimu_experiments.py ...
    The script never overrides either; it only fills them in when unset.
    """
    ext_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if ext_cuda is None and gpu_arg is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_arg)

    n_cpu = available_cpus()
    n_thread = thread_arg if thread_arg else max(1, min(n_cpu, 16))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n_thread))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"),
        "cuda_from_env": ext_cuda is not None,
        "cpu_affinity": n_cpu,
        "torch_threads": n_thread,
    }


# Dataset on the Hugging Face Hub. The pinned revision is the one used for the
# paper: it yields exactly the reported 12,839 sentence pairs, 12,800 groups and
# the 80:10:10 splits of seeds 42–46.
HF_DATASET_ID = "LKAusagi/VietMuong"
HF_DATASET_REVISION = "d129079b0d8bf2c887f41ad879d884d9f3ad15c0"
HF_SENTENCE_FILE = "sentence_pairs.csv"
HF_WORD_FILE = "word_pairs.csv"
DEFAULT_OUT_DIR = "./vimu_runs"

COL_M = "SentenceM"   # Muong
COL_V = "SentenceV"   # Vietnamese

DIRECTIONS = ["m2v", "v2m"]

# T5-family task prefixes. They are part of the model input and are kept in
# Vietnamese exactly as used for the paper's runs.
PREFIX = {
    "m2v": "Dịch tiếng Mường sang tiếng Việt: ",
    "v2m": "Dịch tiếng Việt sang tiếng Mường: ",
}

# Split ratios: (train, val, test)
SPLIT_SCHEMES = {
    "80:10:10": (0.80, 0.10, 0.10),
    "70:10:20": (0.70, 0.10, 0.20),
    "60:10:30": (0.60, 0.10, 0.30),
}

# Language codes. Muong ISO 639-3 = 'mtq'.
NLLB_VI = "vie_Latn"
NLLB_MU = "mtq_Latn"     # new token; embedding initialised from vie_Latn
# mBART-50 is not part of ALL_MODELS (NLLB-200 replaces it). The language-code
# support below is kept so that it can be re-added with one MODEL_REGISTRY line.
MBART_VI = "vi_VN"
MBART_MU = "mu_VN"       # new token; embedding initialised from vi_VN


@dataclass
class ModelSpec:
    key: str
    hf_name: str
    family: str                 # t5 | mt5 | byt5 | mbart | nllb | scratch_transformer
    max_len: int = 256
    use_prefix: bool = True     # T5-family models use a text prefix
    lang_tokens: bool = False   # mBART/NLLB use language-code tokens
    train_bs: int = 8
    eval_bs: int = 8
    grad_accum: int = 2
    lr: float = 1e-4
    optim: str = "adafactor"
    epochs: int = 20
    grad_ckpt: bool = True
    cost: float = 1.0           # relative run-time weight (for ETA only)
    supports_vocab_expand: bool = True
    params_m: Optional[float] = None   # millions of parameters; ONLY used to
                                       # estimate VRAM when several workers share a GPU
    note: str = ""


def estimate_vram_gb(spec: "ModelSpec") -> Optional[float]:
    """Estimate the fixed VRAM of a job (independent of batch size).

    Counts fp32 weights + fp32 gradients + optimizer state. AdamW keeps two
    fp32 moments (8 bytes/parameter); Adafactor factorises the second moment,
    so it is treated as ~0. Adds ~1.5 GB for activations, CUDA context and
    fragmentation. Returns None when the parameter count is unknown.
    """
    if not spec.params_m:
        return None
    p = spec.params_m * 1e6
    opt_bytes = 8.0 if spec.optim.startswith("adamw") and "8bit" not in spec.optim else 0.05
    fixed = p * (4 + 4 + opt_bytes) / 2 ** 30
    return fixed + 1.5


# Settings for an RTX A5000 24 GB (bf16). Effective batch size is 16 for every model.
# max_len: the longest corpus sentence is ~35 syllables, so 96 subwords is ample;
# a larger limit only makes beam search run longer during evaluation.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "vit5-base": ModelSpec(
        key="vit5-base", hf_name="VietAI/vit5-base", family="t5", max_len=96,
        train_bs=16, eval_bs=32, grad_accum=1, lr=1e-4, optim="adafactor",
        epochs=10, grad_ckpt=False, cost=1.0,
        params_m=225.95,
    ),
    "vit5-large": ModelSpec(
        key="vit5-large", hf_name="VietAI/vit5-large", family="t5", max_len=96,
        train_bs=8, eval_bs=16, grad_accum=2, lr=1e-4, optim="adafactor",
        epochs=8, grad_ckpt=False, cost=2.4,
        params_m=791.28,
    ),
    "mt5-base": ModelSpec(
        key="mt5-base", hf_name="google/mt5-base", family="mt5", max_len=96,
        train_bs=8, eval_bs=16, grad_accum=2, lr=1e-3, optim="adafactor",
        epochs=10, grad_ckpt=False, cost=1.6,
        note="mT5 needs a higher LR (1e-3) because of its pure span-corruption pre-training.",
        params_m=582.4,
    ),
    "mt5-large": ModelSpec(
        key="mt5-large", hf_name="google/mt5-large", family="mt5", max_len=96,
        train_bs=4, eval_bs=8, grad_accum=4, lr=5e-4, optim="adafactor",
        epochs=10, grad_ckpt=True, cost=3.2,
        note="~1.2B parameters (~2x mt5-base). Smaller batch + gradient checkpointing "
             "to fit a 24 GB A5000; LR lowered to 5e-4 (vs 1e-3 for mt5-base) "
             "because the larger model is more sensitive to a high LR with only "
             "~10K training sentences. cost=3.2 is an initial ETA estimate that "
             "the script recalibrates after the first finished run.",
        params_m=1230.0,
    ),
    "nllb600": ModelSpec(
        key="nllb600", hf_name="facebook/nllb-200-distilled-600M", family="nllb",
        max_len=96, use_prefix=False, lang_tokens=True,
        train_bs=16, eval_bs=32, grad_accum=1, lr=5e-5, optim="adafactor",
        epochs=8, grad_ckpt=False, cost=1.6,
        note="Larger batch, no gradient checkpointing, 8 epochs. Adafactor, like "
             "every other model in the paper; lr 5e-5.",
        params_m=615.07,
    ),
    "nllb1.3b": ModelSpec(
        key="nllb1.3b", hf_name="facebook/nllb-200-distilled-1.3B", family="nllb",
        max_len=96, use_prefix=False, lang_tokens=True,
        train_bs=4, eval_bs=8, grad_accum=4, lr=3e-5, optim="adafactor",
        epochs=8, grad_ckpt=True, cost=3.2,
        note="~1.3B parameters (~2.1x nllb600). Adafactor keeps the fixed memory "
             "at ~10.3 GB (fp32 weights + gradients; the factorised second moment "
             "is ~0.09 GB), whereas AdamW would need ~20.4 GB before any batch is "
             "loaded. cost=3.2 is an initial ETA estimate, recalibrated after the "
             "first finished run.",
        params_m=1370.0,
    ),
    "byt5-small": ModelSpec(
        key="byt5-small", hf_name="google/byt5-small", family="byt5",
        max_len=320,                     # byte-level: 35 syllables ~ 175 bytes
        train_bs=8, eval_bs=16, grad_accum=2, lr=5e-4, optim="adafactor",
        epochs=12, grad_ckpt=False, cost=2.0,
        supports_vocab_expand=False,
        note="Hypothesis: high cognate density favours byte-level models.",
        params_m=299.64,
    ),
    "byt5-large": ModelSpec(
        key="byt5-large", hf_name="google/byt5-large", family="byt5",
        max_len=320,                     # same byte-level length as byt5-small
        train_bs=2, eval_bs=4, grad_accum=8, lr=2e-4, optim="adafactor",
        epochs=12, grad_ckpt=True, cost=5.0,
        supports_vocab_expand=False,
        note="~1.23B parameters (~4.1x byt5-small). Smallest batch in the registry "
             "because it combines the largest model with the longest (byte) "
             "sequences (max_len=320 vs 96 for subword models). grad_ckpt=True and "
             "train_bs=2 x grad_accum=8 keep the effective batch at 16, as for "
             "byt5-small. LR lowered to 2e-4. cost=5.0 is an initial ETA estimate, "
             "recalibrated after the first finished run.",
        params_m=1230.0,
    ),
    "transformer-small": ModelSpec(
        key="transformer-small", hf_name="__scratch_transformer__",
        family="scratch_transformer", max_len=96,
        train_bs=64, eval_bs=64, grad_accum=1, lr=3e-4, optim="adamw_torch",
        epochs=40, grad_ckpt=False, cost=0.22,
        note="Standard low-resource baseline: Transformer 6L-6L d512, own tokenizer (not used in the paper).",
        params_m=60.0,
    ),
}

# Data strategies -------------------------------------------------------------
STRATEGIES = [
    "base",          # sentences only (paper: sentence-only)
    "word",          # + all dictionary pairs in TRAIN (paper: word mixing)
    "token",         # vocabulary expansion, random embedding init (paper: token)
    "token_avg",     # vocabulary expansion, init = mean of subword embeddings (paper: token_avg)
    "word_token",    # word + token_avg
    "ctrl_shuffle",  # control: lexical pairs with shuffled targets (paper: shuffled targets)
    "ctrl_dup",      # control: duplicated training sentences (paper: duplicated sentences)
    "word_p25", "word_p50", "word_p75",   # scaling curve
]

STRATEGY_META = {
    "base":         dict(word=0.0,  expand=None,  shuffle=False, dup=False),
    "word":         dict(word=1.0,  expand=None,  shuffle=False, dup=False),
    "token":        dict(word=1.0,  expand="rand", shuffle=False, dup=False),
    "token_avg":    dict(word=1.0,  expand="avg",  shuffle=False, dup=False),
    "word_token":   dict(word=1.0,  expand="avg",  shuffle=False, dup=False),
    "ctrl_shuffle": dict(word=1.0,  expand=None,  shuffle=True,  dup=False),
    "ctrl_dup":     dict(word=0.0,  expand=None,  shuffle=False, dup=True),
    "word_p25":     dict(word=0.25, expand=None,  shuffle=False, dup=False),
    "word_p50":     dict(word=0.50, expand=None,  shuffle=False, dup=False),
    "word_p75":     dict(word=0.75, expand=None,  shuffle=False, dup=False),
}

# ==============================================================================
# EXPERIMENT DESIGN
# ==============================================================================
# Every experiment is a FULL factorial grid: within one experiment every cell
# has the same seeds, models, strategies and both translation directions.
# When the GPU budget is short, a WHOLE experiment is dropped, never single cells.
#
#
#   E1  Main table        : all pre-trained models x {base, word} x 2 directions  (Table II)
#   E5  Seed variance     : 1 model x {base, word} x 2 directions x many seeds
#                           (skipped automatically when --seeds has >= 3 values,
#                            which is the default and the paper setting)
#   E2  Pre-training role : same architecture, random init (NOT used in the paper)
#   E3  Strategy ablation : ViT5-base x all data strategies  (Table III, Sec. V-C)
#   E4  Data contraction  : ViT5-base x {base, word} x other split ratios
#                           (70:10:20, 60:10:30; not reported in the paper's tables)
# E0 are the CPU baselines (copy, rule-based); they are cheap and always run.
# Default model set of the main table.
#
# The three large variants (mt5-large, nllb1.3b, byt5-large) are part of E1 and
# run the same {base, word} grid as every other model, giving base<->large pairs
# for mT5, NLLB and ByT5 in the same table.
#
# Budget note: E1 is 8 models x 2 strategies x 2 directions x 3 seeds = 96 GPU
# runs; the large models dominate the cost (see `cost` in MODEL_REGISTRY).
# To run fewer models (the table stays a full grid), pass
# --models vit5-base vit5-large mt5-base nllb600 byt5-small.
ALL_MODELS = ["vit5-base", "vit5-large",
              "mt5-base", "mt5-large",
              "nllb600", "nllb1.3b",
              "byt5-small", "byt5-large"]

# E2_scratch (training from random weights) is NOT part of the paper's design.
# The supporting code is kept and can be re-enabled by adding "E2_scratch"
# back to EXPERIMENTS below
# (--scratch_models, --include_transformer_scratch, transformer-small,
# adjusted LR/patience for from_scratch).
EXPERIMENTS = ["E1_main", "E5_seed_variance",
               "E3_strategy", "E4_contraction"]

EXPERIMENT_TITLE = {
    "E1_main": "Main table: models x {base, word}",
    "E5_seed_variance": "Seed variance (only when fewer than 3 seeds are used)",
    "E2_scratch": "Role of pre-training: same architecture, random init (not in the paper)",
    "E3_strategy": "Data-strategy ablation (ViT5-base)",
    "E4_contraction": "Training-data contraction (70:10:20, 60:10:30; not in the paper)",
}

# ==============================================================================
# SECTION 1 — Text utilities: normalisation, tones, cognates
# ==============================================================================

TONE_MARKS = {
    "̀": "huyen",   # à
    "́": "sac",     # á
    "̃": "nga",     # ã
    "̉": "hoi",     # ả
    "̣": "nang",    # ạ
}
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """Light normalisation: NFC, lowercase, strip punctuation, collapse spaces."""
    s = unicodedata.normalize("NFC", str(s)).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def strip_tones(s: str) -> str:
    """Remove all tone marks (keep letter diacritics: ă â ê ô ơ ư đ)."""
    d = unicodedata.normalize("NFD", str(s))
    return unicodedata.normalize("NFC", "".join(c for c in d if c not in TONE_MARKS))


def syllable_tone(syl: str) -> str:
    """Return the tone name of a syllable ('ngang' when unmarked)."""
    d = unicodedata.normalize("NFD", str(syl))
    for c in d:
        if c in TONE_MARKS:
            return TONE_MARKS[c]
    return "ngang"


def tone_sequence(text: str) -> List[str]:
    return [syllable_tone(t) for t in norm_text(text).split()]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_edit_sim(a: str, b: str) -> float:
    a, b = norm_text(a), norm_text(b)
    if not a and not b:
        return 1.0
    m = max(len(a), len(b)) or 1
    return 1.0 - levenshtein(a, b) / m


def cognate_density(src: str, ref: str, thr: float = 0.66) -> float:
    """Fraction of position-aligned syllables with character similarity >= thr.

    Valid because Vietnamese–Muong pairs are almost monotone (length ratio 0.97).
    """
    s, r = norm_text(src).split(), norm_text(ref).split()
    if not s or not r:
        return 0.0
    n = min(len(s), len(r))
    hit = sum(1 for i in range(n) if norm_edit_sim(s[i], r[i]) >= thr)
    return hit / max(len(s), len(r))


def shingles(tokens: Sequence[str], k: int = 3) -> set:
    if len(tokens) < k:
        return {" ".join(tokens)}
    return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


# ==============================================================================
# SECTION 2 — Data loading, deduplication, grouping, leakage-free splits
# ==============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def resolve_data_files(sentence_file: Optional[str], word_file: Optional[str],
                       revision: str = HF_DATASET_REVISION) -> Tuple[str, str]:
    """Return local paths of the sentence-pair and lexical-pair CSV files.

    Files given on the command line are used as they are; missing ones are
    downloaded from the Hugging Face Hub (and cached by huggingface_hub).
    """
    if sentence_file and word_file:
        return sentence_file, word_file
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> str:
        return hf_hub_download(repo_id=HF_DATASET_ID, filename=name,
                               repo_type="dataset", revision=revision)
    return sentence_file or fetch(HF_SENTENCE_FILE), word_file or fetch(HF_WORD_FILE)


def load_frames(sentence_file: str, word_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_s = pd.read_csv(sentence_file)
    df_w = pd.read_csv(word_file)
    ren = {}
    for c in df_w.columns:
        if c.lower() in ("wordm", "word_m", "m", "muong"):
            ren[c] = COL_M
        if c.lower() in ("wordv", "word_v", "v", "viet", "vietnamese"):
            ren[c] = COL_V
    df_w = df_w.rename(columns=ren)
    for df, name in ((df_s, "sentence"), (df_w, "word")):
        missing = {COL_M, COL_V} - set(df.columns)
        if missing:
            raise ValueError(f"File {name} is missing columns {missing}. Available columns: {list(df.columns)}")
    df_s = df_s[[COL_M, COL_V]].dropna().astype(str)
    df_w = df_w[[COL_M, COL_V]].dropna().astype(str)
    df_s[COL_M] = df_s[COL_M].str.strip()
    df_s[COL_V] = df_s[COL_V].str.strip()
    df_w[COL_M] = df_w[COL_M].str.strip()
    df_w[COL_V] = df_w[COL_V].str.strip()
    df_s = df_s[(df_s[COL_M] != "") & (df_s[COL_V] != "")]
    df_w = df_w[(df_w[COL_M] != "") & (df_w[COL_V] != "")]
    return df_s.reset_index(drop=True), df_w.reset_index(drop=True)


def dedup_and_group(df: pd.DataFrame,
                    jaccard_thr: float = 0.8,
                    group_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Exact deduplication + grouping of near-duplicates into 'group'.

    Splitting by 'group' keeps two near-identical sentences on the same side.
    """
    stats: Dict[str, int] = {"raw": len(df)}

    key = (df[COL_M].map(norm_text) + " ||| " + df[COL_V].map(norm_text))
    df = df.assign(_key=key).drop_duplicates("_key").reset_index(drop=True)
    stats["after_exact_dedup"] = len(df)

    if group_col is not None and group_col in df.columns:
        codes = pd.factorize(df[group_col])[0]
        df = df.assign(group=codes).drop(columns=["_key"])
        stats["n_groups"] = int(df["group"].nunique())
        return df, stats

    # --- near-duplicates via an inverted index over 3-shingles of the Muong side ---
    toks = [norm_text(s).split() for s in df[COL_M]]
    sh = [shingles(t, 3) for t in toks]
    index: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        for g in s:
            index[g].append(i)

    uf = UnionFind(len(df))
    for _, bucket in index.items():
        if len(bucket) < 2 or len(bucket) > 200:      # skip overly common shingles
            continue
        for a, b in itertools.combinations(bucket, 2):
            if uf.find(a) == uf.find(b):
                continue
            inter = len(sh[a] & sh[b])
            union = len(sh[a] | sh[b]) or 1
            if inter / union >= jaccard_thr:
                uf.union(a, b)

    roots = np.array([uf.find(i) for i in range(len(df))])
    df = df.assign(group=pd.factorize(roots)[0]).drop(columns=["_key"])
    stats["n_groups"] = int(df["group"].nunique())
    stats["near_dup_merged"] = stats["after_exact_dedup"] - stats["n_groups"]
    return df, stats


def grouped_split(df: pd.DataFrame, ratios: Tuple[float, float, float], seed: int
                  ) -> Dict[str, pd.DataFrame]:
    """Split by GROUP: no group is ever divided between two splits."""
    groups = df["group"].unique().tolist()
    rng = random.Random(seed)
    rng.shuffle(groups)

    sizes = df.groupby("group").size().to_dict()
    total = len(df)
    tr_t, va_t = ratios[0] * total, (ratios[0] + ratios[1]) * total

    assign, acc = {}, 0
    for g in groups:
        if acc < tr_t:
            assign[g] = "train"
        elif acc < va_t:
            assign[g] = "val"
        else:
            assign[g] = "test"
        acc += sizes[g]

    part = df["group"].map(assign)
    return {s: df[part == s].drop(columns=["group"]).reset_index(drop=True)
            for s in ("train", "val", "test")}


def filter_word_pairs_leaking_into_test(df_w: pd.DataFrame, df_test: pd.DataFrame
                                        ) -> Tuple[pd.DataFrame, int]:
    """Drop dictionary pairs whose Muong side appears in the test set.

    Gives a strictly clean ablation (--exclude_test_words; not used in the paper).
    """
    test_ngrams = set()
    for s in df_test[COL_M]:
        t = norm_text(s).split()
        for n in (1, 2, 3):
            for i in range(len(t) - n + 1):
                test_ngrams.add(" ".join(t[i:i + n]))
    keep = [norm_text(w) not in test_ngrams for w in df_w[COL_M]]
    removed = int(len(keep) - sum(keep))
    return df_w[pd.Series(keep, index=df_w.index)].reset_index(drop=True), removed


def build_train_frame(df_train: pd.DataFrame, df_word: pd.DataFrame,
                      strategy: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply a data strategy to TRAIN only (val/test are never modified)."""
    meta = STRATEGY_META[strategy]
    rng = np.random.RandomState(seed)
    parts = [df_train]
    info: Dict[str, Any] = {"n_sentence_train": len(df_train), "n_word_added": 0}

    frac = meta["word"]
    if frac > 0 and len(df_word) > 0:
        n = int(round(frac * len(df_word)))
        idx = rng.permutation(len(df_word))[:n]
        wp = df_word.iloc[idx].copy().reset_index(drop=True)
        if meta["shuffle"]:
            # CONTROL: keep the token distribution but break the correct mapping
            perm = rng.permutation(len(wp))
            wp[COL_V] = wp[COL_V].values[perm]
            info["control"] = "shuffled_word_targets"
        parts.append(wp[[COL_M, COL_V]])
        info["n_word_added"] = len(wp)

    if meta["dup"]:
        # CONTROL: add instances WITHOUT adding lexical knowledge.
        # The number of extra instances matches the "word" strategy exactly
        # (round(frac_word * len(df_word)) = 15,377 for the full dictionary), so
        # word mixing and ctrl_dup both have 25,649 training instances. Sampling
        # is with replacement (rng.randint), so no upper bound by len(df_train)
        # is needed.
        frac_word_target = STRATEGY_META["word"]["word"]
        n_extra = int(round(frac_word_target * len(df_word)))
        idx = rng.randint(0, len(df_train), size=n_extra)
        parts.append(df_train.iloc[idx][[COL_M, COL_V]])
        info["control"] = "duplicated_sentences"
        info["n_dup_added"] = int(n_extra)
        info["n_dup_matched_to_word_strategy"] = True

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    info["n_train_total"] = len(out)
    return out, info


# ==============================================================================
# SECTION 3 — Metrics
# ==============================================================================

def _sacre():
    import sacrebleu
    return sacrebleu


def compute_all_metrics(sources: List[str], hyps: List[str], refs: List[str]
                        ) -> Dict[str, Any]:
    sb = _sacre()
    bleu_m = sb.metrics.BLEU()
    chrf_m = sb.metrics.CHRF(word_order=2)   # chrF++
    ter_m = sb.metrics.TER()

    bleu = bleu_m.corpus_score(hyps, [refs])
    chrf = chrf_m.corpus_score(hyps, [refs])
    ter = ter_m.corpus_score(hyps, [refs])

    res: Dict[str, Any] = {
        "bleu": round(bleu.score, 4),
        "chrf2": round(chrf.score, 4),
        "ter": round(ter.score, 4),
        "bleu_signature": str(bleu_m.get_signature()),
        "chrf_signature": str(chrf_m.get_signature()),
        "n_test": len(hyps),
    }

    # --- copy rate: is the system just copying the source? ---
    exact_copy = np.mean([norm_text(h) == norm_text(s) for h, s in zip(hyps, sources)])
    res["copy_rate_exact"] = round(float(exact_copy) * 100, 3)
    res["src_hyp_charsim"] = round(float(np.mean(
        [norm_edit_sim(s, h) for s, h in zip(sources, hyps)])) * 100, 3)

    # --- length ratio ---
    lh = np.array([len(norm_text(h).split()) for h in hyps], dtype=float)
    lr = np.array([len(norm_text(r).split()) for r in refs], dtype=float)
    res["len_ratio_hyp_ref"] = round(float(lh.sum() / max(lr.sum(), 1)), 4)

    # --- TONE ACCURACY (specific to Vietnamese–Muong) ---
    tot = hit = 0
    conf: Counter = Counter()
    for h, r in zip(hyps, refs):
        th, tr = tone_sequence(h), tone_sequence(r)
        for a, b in zip(th, tr):
            tot += 1
            if a == b:
                hit += 1
            else:
                conf[f"{b}->{a}"] += 1
    res["tone_accuracy"] = round(100.0 * hit / max(tot, 1), 3)
    res["tone_confusions_top10"] = conf.most_common(10)

    # --- accuracy with tones REMOVED: separates consonant/rhyme errors from tone errors ---
    hyps_nt = [strip_tones(h) for h in hyps]
    refs_nt = [strip_tones(r) for r in refs]
    res["bleu_no_tone"] = round(bleu_m.corpus_score(hyps_nt, [refs_nt]).score, 4)
    res["tone_induced_bleu_gap"] = round(res["bleu_no_tone"] - res["bleu"], 4)

    # --- BLEU/chrF by COGNATE-density band (paper: surface-overlap bands, Table IV) ---
    dens = np.array([cognate_density(s, r) for s, r in zip(sources, refs)])
    if len(dens) >= 30:
        q1, q2 = np.quantile(dens, [1 / 3, 2 / 3])
        buckets = {"low": dens <= q1, "mid": (dens > q1) & (dens <= q2), "high": dens > q2}
        res["cognate_density_mean"] = round(float(dens.mean()), 4)
        for name, mask in buckets.items():
            if mask.sum() < 5:
                continue
            hh = [h for h, m in zip(hyps, mask) if m]
            rr = [r for r, m in zip(refs, mask) if m]
            res[f"bleu_cognate_{name}"] = round(bleu_m.corpus_score(hh, [rr]).score, 3)
            res[f"chrf2_cognate_{name}"] = round(chrf_m.corpus_score(hh, [rr]).score, 3)
            res[f"n_cognate_{name}"] = int(mask.sum())
    return res


def paired_bootstrap(hyps_a: List[str], hyps_b: List[str], refs: List[str],
                     n_samples: int = 1000, seed: int = 12345) -> Dict[str, float]:
    """Paired bootstrap resampling: is system A really better than system B?

    Implemented on per-sentence n-gram STATISTICS (Koehn 2004): each resample
    sums the statistic vectors and computes BLEU once instead of re-tokenising
    the corpus, which is about three orders of magnitude faster.
    """
    sb = _sacre()
    bleu_m = sb.metrics.BLEU(effective_order=True)
    refs_l = [refs]
    sa = np.asarray(bleu_m._extract_corpus_statistics(hyps_a, refs_l), dtype=np.float64)
    sbb = np.asarray(bleu_m._extract_corpus_statistics(hyps_b, refs_l), dtype=np.float64)

    def score(stats_sum: np.ndarray) -> float:
        return bleu_m._compute_score_from_stats(stats_sum.tolist()).score

    base_a = score(sa.sum(0))
    base_b = score(sbb.sum(0))

    rng = np.random.RandomState(seed)
    n = len(refs)
    deltas = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        idx = rng.randint(0, n, size=n)
        deltas[k] = score(sa[idx].sum(0)) - score(sbb[idx].sum(0))
    wins = int((deltas > 0).sum())
    return {
        "bleu_a": round(base_a, 4),
        "bleu_b": round(base_b, 4),
        "delta": round(base_a - base_b, 4),
        "p_value": round(1.0 - wins / n_samples, 5),
        "ci95_low": round(float(np.quantile(deltas, 0.025)), 4),
        "ci95_high": round(float(np.quantile(deltas, 0.975)), 4),
        "n_samples": n_samples,
    }


# ==============================================================================
# SECTION 4 — Baselines that need no GPU
# ==============================================================================

def baseline_copy(sources: List[str]) -> List[str]:
    """IDENTITY: output the source unchanged. Mandatory baseline for related languages."""
    return [str(s).strip() for s in sources]


class RuleBasedTranslator:
    """P(M|V) from co-occurrence counts (Sec. IV-A of the paper) + longest-match segmentation."""

    def __init__(self, df_word: pd.DataFrame, src_col: str, tgt_col: str):
        self.table: Dict[str, Counter] = defaultdict(Counter)
        self.max_n = 1
        for s, t in zip(df_word[src_col], df_word[tgt_col]):
            k = norm_text(s)
            if not k:
                continue
            self.table[k][str(t).strip()] += 1
            self.max_n = max(self.max_n, len(k.split()))
        self.best = {k: c.most_common(1)[0][0] for k, c in self.table.items()}
        self.max_n = min(self.max_n, 4)

    def translate(self, sent: str) -> str:
        toks = norm_text(sent).split()
        out, i = [], 0
        while i < len(toks):
            hit = None
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                cand = " ".join(toks[i:i + n])
                if cand in self.best:
                    hit = (self.best[cand], n)
                    break
            if hit:
                out.append(hit[0])
                i += hit[1]
            else:
                out.append(toks[i])          # OOV: keep as is
                i += 1
        return " ".join(out)

    def coverage(self, sents: Iterable[str]) -> float:
        tot = hit = 0
        for s in sents:
            for t in norm_text(s).split():
                tot += 1
                hit += int(t in self.best)
        return 100.0 * hit / max(tot, 1)


# ==============================================================================
# SECTION 5 — Models / tokenizers (torch is imported only when a GPU job needs it)
# ==============================================================================

def _torch():
    import torch
    return torch


def pick_precision() -> Dict[str, bool]:
    """bf16 when the GPU supports it. Never fp16 for the T5 family (NaN-prone)."""
    torch = _torch()
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"bf16": True, "fp16": False}
    return {"bf16": False, "fp16": False}   # slow fp32 is better than NaN


def add_language_token(tokenizer, model, new_code: str, donor_code: str, family: str):
    """Add the Muong language code to NLLB/mBART, initialising its embedding from Vietnamese."""
    torch = _torch()
    if new_code in tokenizer.get_vocab():
        return tokenizer, model
    extra = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    tokenizer.add_special_tokens({"additional_special_tokens": extra + [new_code]})
    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(new_code)
    donor_id = tokenizer.convert_tokens_to_ids(donor_code)
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        emb[new_id] = emb[donor_id].clone()
        out = model.get_output_embeddings()
        if out is not None and out.weight.shape[0] > new_id and out.weight is not emb:
            out.weight[new_id] = out.weight[donor_id].clone()

    for attr in ("lang_code_to_id", "id_to_lang_code"):
        if hasattr(tokenizer, attr):
            d = getattr(tokenizer, attr)
            if attr == "lang_code_to_id":
                d[new_code] = new_id
            else:
                d[new_id] = new_code
    if hasattr(tokenizer, "_additional_special_tokens"):
        pass
    return tokenizer, model


def expand_vocab(tokenizer, model, texts: Iterable[str], mode: str) -> Dict[str, Any]:
    """Expand the tokenizer with new syllables via add_tokens (not used by the paper;
    see expand_vocab_word_boundary).

    mode='rand' : random initialisation
    mode='avg'  : initialise with the MEAN embedding of the token's own subwords
                  (Hewitt 2021 / WECHSEL / FOCUS).
    """
    torch = _torch()
    vocab = tokenizer.get_vocab()
    syls = set()
    for t in texts:
        for w in norm_text(t).split():
            syls.add(w)
    new_tokens = [s for s in sorted(syls)
                  if s not in vocab and f"▁{s}" not in vocab and f" {s}" not in vocab]
    if not new_tokens:
        return {"n_new_tokens": 0, "init": mode}

    old_size = model.get_input_embeddings().weight.shape[0]
    pieces = {t: tokenizer.tokenize(t) for t in new_tokens} if mode == "avg" else {}
    n_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if mode == "avg":
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            out = model.get_output_embeddings()
            for t in new_tokens:
                sub = [tokenizer.convert_tokens_to_ids(p) for p in pieces.get(t, [])]
                sub = [i for i in sub if isinstance(i, int) and 0 <= i < old_size]
                if not sub:
                    continue
                new_id = tokenizer.convert_tokens_to_ids(t)
                v = emb[sub].mean(dim=0)
                emb[new_id] = v
                if out is not None and out.weight is not emb and out.weight.shape[0] > new_id:
                    out.weight[new_id] = out.weight[sub].mean(dim=0)
    return {"n_new_tokens": int(n_added), "init": mode, "vocab_size": len(tokenizer)}


def build_scratch_transformer(train_texts: List[str], out_dir: str, vocab_size: int = 8000):
    """Transformer 6L-6L d512 + its own SentencePiece tokenizer, random init (not in the paper)."""
    import sentencepiece as spm
    from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    corpus = Path(out_dir) / "spm_corpus.txt"
    corpus.write_text("\n".join(train_texts), encoding="utf-8")
    prefix = str(Path(out_dir) / "spm")
    if not Path(prefix + ".model").exists():
        # Shrink the vocabulary automatically if the corpus is too small for SentencePiece.
        for vs in [vocab_size, 4000, 2000, 1000, 500, 250, 120, 64]:
            if vs > vocab_size:
                continue
            try:
                spm.SentencePieceTrainer.train(
                    input=str(corpus), model_prefix=prefix, vocab_size=vs,
                    character_coverage=1.0, model_type="unigram",
                    pad_id=0, eos_id=1, unk_id=2, bos_id=-1,
                    user_defined_symbols=[f"<extra_id_{i}>" for i in range(10)],
                    minloglevel=2,
                )
                break
            except RuntimeError as e:
                m = re.search(r"set it to a value <= (\d+)", str(e))
                if m:
                    vocab_size = min(vocab_size, int(m.group(1)))
                if vs == 64:
                    raise
    tok = T5TokenizerFast(vocab_file=prefix + ".model", extra_ids=0,
                          model_max_length=256)
    cfg = T5Config(
        vocab_size=len(tok), d_model=512, d_ff=2048, d_kv=64,
        num_layers=6, num_decoder_layers=6, num_heads=8,
        dropout_rate=0.3, feed_forward_proj="gated-gelu",
        decoder_start_token_id=tok.pad_token_id, pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        tie_word_embeddings=True,   # avoids a "missing keys" warning when reloading the best checkpoint
    )
    model = T5ForConditionalGeneration(cfg)
    return tok, model


def new_syllables_like_old(tokenizer, texts: Iterable[str]) -> List[str]:
    """New syllables, selected with the same criterion as expand_vocab()."""
    vocab = tokenizer.get_vocab()
    syls = set()
    for t in texts:
        for w in norm_text(t).split():
            syls.add(w)
    return [s for s in sorted(syls)
            if s not in vocab and f"▁{s}" not in vocab and f" {s}" not in vocab]


def build_word_boundary_tokenizer(tokenizer, texts: Iterable[str], out_path: Path):
    """Expand the ViT5 tokenizer WITHOUT splitting existing words.

    ViT5's tokenizer.json is a BPE model with Metaspace pre-tokenisation (split on
    whitespace, prefix every word with '▁'). Unlike tokenizer.add_tokens() (added
    tokens match substrings ANYWHERE, before BPE, so 'này' becomes 'nà'+'y'):
      * each new syllable s is added as ONE BPE vocabulary entry '▁'+s, with ids
        continuing AFTER the <extra_id_*> sentinels (36096, 36097, ...): every old
        id is unchanged and resize_token_embeddings works directly;
      * NO merges are added, so no merge sequence can ever produce '▁'+s inside
        a longer word;
      * ignore_merges=True: a whole word (the span between two spaces, with '▁')
        that exists in the vocabulary is mapped directly to a single token.
        => '▁'+s is used only when the WHOLE word equals s. The side effect of
        ignore_merges on existing words is checked in tests/ (it must be zero on
        the whole corpus + dictionary).
    Returns (new_tokenizer, list_of_(syllable, piece), info).
    """
    cand = new_syllables_like_old(tokenizer, texts)
    vocab = tokenizer.get_vocab()
    norm = tokenizer.backend_tokenizer.normalizer
    pieces: List[Tuple[str, str]] = []
    seen = set()
    skipped = []
    for s in cand:
        ns = norm.normalize_str(s) if norm is not None else s
        p = "▁" + ns
        if p in vocab or p in seen:
            skipped.append(s)
            continue
        seen.add(p)
        pieces.append((s, p))
    tj = json.loads(tokenizer.backend_tokenizer.to_str())
    if tj["model"]["type"] != "BPE":
        raise RuntimeError(f"build_word_boundary_tokenizer: expected BPE, got {tj['model']['type']}")
    v = tj["model"]["vocab"]
    old_n = len(tokenizer)
    if max(v.values()) + 1 != old_n or len(v) != old_n:
        raise RuntimeError(f"BPE vocab ({len(v)}, max id {max(v.values())}) does not match len(tokenizer)={old_n}")
    for i, (_, p) in enumerate(pieces):
        v[p] = old_n + i
    tj["model"]["ignore_merges"] = True
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    new_tok = load_t5_tokenizer_v4compat(getattr(tokenizer, "_vimu_hf_name", "VietAI/vit5-base"),
                                         tokenizer_file=str(out_path))
    if len(new_tok) != old_n + len(pieces):
        raise RuntimeError(f"len(new_tok)={len(new_tok)} != {old_n}+{len(pieces)}")
    info = {"n_candidates": len(cand), "n_new_tokens": len(pieces),
            "n_skipped_after_normalization": len(skipped),
            "new_id_range": [old_n, old_n + len(pieces) - 1] if pieces else None}
    return new_tok, pieces, info


def expand_vocab_word_boundary(tokenizer, model, texts: Iterable[str], mode: str,
                               work_dir: Path):
    """Vocabulary expansion used in the paper (token / token_avg). Returns (new_tokenizer, info).

    mode='rand': new embedding rows ~ N(0, initializer_factor*1.0), i.e.
                 resize_token_embeddings without mean_resizing.
    mode='avg' : new row = MEAN embedding of the pieces the ORIGINAL tokenizer
                 produces for the syllable at word start (tokenizer(s), with '▁').
    ViT5 ties its embeddings to lm_head (tie_word_embeddings=True), so one matrix
    is written; an untied lm_head is handled the same way.
    """
    torch = _torch()
    texts = list(texts)
    new_tok, pieces, info = build_word_boundary_tokenizer(
        tokenizer, texts, Path(work_dir) / "tokenizer_vx" / "tokenizer.json")
    emb_mod = model.get_input_embeddings()
    old_rows = emb_mod.weight.shape[0]
    old_n = len(tokenizer)
    if old_rows != old_n:
        raise RuntimeError(f"embedding rows {old_rows} != len(tokenizer) {old_n}")
    out_mod = model.get_output_embeddings()
    tied = out_mod is not None and out_mod.weight.data_ptr() == emb_mod.weight.data_ptr()
    orig_in = emb_mod.weight.detach().clone()
    orig_out = None if (out_mod is None or tied) else out_mod.weight.detach().clone()
    try:
        model.resize_token_embeddings(len(new_tok), mean_resizing=False)
    except TypeError:
        model.resize_token_embeddings(len(new_tok))
    emb = model.get_input_embeddings().weight
    out_w = model.get_output_embeddings().weight if model.get_output_embeddings() is not None else None
    tied_after = out_w is not None and out_w.data_ptr() == emb.data_ptr()
    std = float(getattr(model.config, "initializer_factor", 1.0)) * 1.0
    n_unk_in_avg = 0
    with torch.no_grad():
        emb[:old_rows].copy_(orig_in.to(emb.device, emb.dtype))
        if orig_out is not None and out_w is not None and not tied_after:
            out_w[:old_rows].copy_(orig_out.to(out_w.device, out_w.dtype))
        for i, (s, _) in enumerate(pieces):
            nid = old_n + i
            if mode == "avg":
                sub = tokenizer(s, add_special_tokens=False)["input_ids"]
                sub = [j for j in sub if 0 <= j < old_rows]
                n_unk_in_avg += int(tokenizer.unk_token_id in sub)
                if sub:
                    emb[nid] = orig_in[sub].mean(dim=0).to(emb.device, emb.dtype)
                    if orig_out is not None and out_w is not None and not tied_after:
                        out_w[nid] = orig_out[sub].mean(dim=0).to(out_w.device, out_w.dtype)
                    continue
            emb[nid] = torch.randn(emb.shape[1], device=emb.device, dtype=emb.dtype) * std
            if orig_out is not None and out_w is not None and not tied_after:
                out_w[nid] = torch.randn(out_w.shape[1], device=out_w.device,
                                         dtype=out_w.dtype) * std
    info.update({"init": mode, "vocab_size": len(new_tok),
                 "method": "spm_word_boundary",
                 "impl": "BPE vocab append '▁'+syllable after sentinels + "
                         "ignore_merges=True (whole-word match only); old ids/rows unchanged",
                 "tied_embeddings": bool(tied_after)})
    if mode == "rand":
        info["rand_std"] = std
    else:
        info["n_avg_pieces_with_unk"] = n_unk_in_avg
    return new_tok, info


def load_t5_tokenizer_v4compat(hf_name: str, tokenizer_file: Optional[str] = None):
    """Load the ViT5 tokenizer directly from its tokenizer.json.

    The tokenizer.json is used AS IS (BPE model, Metaspace, TemplateProcessing
    appending </s>), with extra_ids as special tokens and
    clean_up_tokenization_spaces=True (ViT5's tokenizer_config.json does not set
    it). transformers 5.x would rebuild T5Tokenizer as a Unigram model from the
    vocab, which breaks ViT5's BPE tokenizer.json, so a PreTrainedTokenizerFast is
    built directly from the file instead.
    """
    import transformers
    from transformers import AutoTokenizer
    if int(transformers.__version__.split(".")[0]) < 5 and tokenizer_file is None:
        return AutoTokenizer.from_pretrained(hf_name)
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    tj = tokenizer_file or hf_hub_download(hf_name, "tokenizer.json")
    cfg = json.loads(Path(hf_hub_download(hf_name, "tokenizer_config.json")
                          ).read_text(encoding="utf-8"))
    n_extra = int(cfg.get("extra_ids", 0) or 0)
    extra = list(cfg.get("additional_special_tokens") or
                 [f"<extra_id_{i}>" for i in range(n_extra)])
    tok = PreTrainedTokenizerFast(
        tokenizer_file=tj,
        eos_token=cfg.get("eos_token", "</s>"),
        unk_token=cfg.get("unk_token", "<unk>"),
        pad_token=cfg.get("pad_token", "<pad>"),
        additional_special_tokens=extra,
        clean_up_tokenization_spaces=cfg.get("clean_up_tokenization_spaces", True),
        model_max_length=int(1e30),
    )
    # transformers 5 SKIPS clean_up for BPE by default; enable it so decoded text
    # has no space before punctuation (" ,").
    tok.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True
    tok._vimu_v4compat = True
    tok._vimu_hf_name = hf_name
    return tok


def load_model_and_tokenizer(spec: ModelSpec, from_scratch: bool, direction: str,
                             train_texts: List[str], work_dir: str):
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    if spec.family == "scratch_transformer":
        tok, model = build_scratch_transformer(train_texts, work_dir)
        return tok, model, {"scratch": True, "arch": "T5-6L6L-d512"}

    tok_kwargs: Dict[str, Any] = {}
    if spec.family == "nllb":
        src = NLLB_MU if direction == "m2v" else NLLB_VI
        tgt = NLLB_VI if direction == "m2v" else NLLB_MU
        tok_kwargs = {"src_lang": NLLB_VI, "tgt_lang": NLLB_VI}   # temporary; set after adding the token
    elif spec.family == "mbart":
        tok_kwargs = {"src_lang": MBART_VI, "tgt_lang": MBART_VI}

    if spec.family == "t5":
        tokenizer = load_t5_tokenizer_v4compat(spec.hf_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_name, **tok_kwargs)
        if spec.family == "nllb":
            # transformers 5 sets clean_up_tokenization_spaces=False for NLLB (its
            # tokenizer_config does not set the key) and also skips clean-up for
            # BPE. Enable both so decoded text has no space before punctuation
            # (" ," / " ?"); encoding is not affected.
            tokenizer.clean_up_tokenization_spaces = True
            tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True

    if from_scratch:
        cfg = AutoConfig.from_pretrained(spec.hf_name)
        cfg.dropout_rate = getattr(cfg, "dropout_rate", 0.1)
        model = AutoModelForSeq2SeqLM.from_config(cfg)
        info = {"scratch": True}
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(spec.hf_name)
        info = {"scratch": False}

    if spec.family == "nllb":
        tokenizer, model = add_language_token(tokenizer, model, NLLB_MU, NLLB_VI, "nllb")
        info["lang_token"] = NLLB_MU
    elif spec.family == "mbart":
        tokenizer, model = add_language_token(tokenizer, model, MBART_MU, MBART_VI, "mbart")
        info["lang_token"] = MBART_MU

    # ---- Fix for "You have to specify either decoder_input_ids or
    # decoder_inputs_embeds" with NLLB (M2M100ForConditionalGeneration) ----
    # label_smoothing_factor > 0 makes Trainer.compute_loss() REMOVE "labels"
    # from the batch before calling model(**inputs) (it computes the smoothed
    # loss itself). DataCollatorForSeq2Seq compensates by precomputing
    # decoder_input_ids and adding them to the batch -- BUT only when the model
    # has
    # `prepare_decoder_input_ids_from_labels`. T5ForConditionalGeneration
    # (vit5/mt5/byt5) has it; M2M100ForConditionalGeneration (nllb)
    # does NOT -- so for NLLB decoder_input_ids are never created, and once
    # "labels" are removed forward() cannot derive decoder_input_ids -> ValueError.
    # The standard shift-right is attached to EVERY model that lacks it, so all
    # models behave the same and label smoothing does not have to be disabled
    # for NLLB alone.
    if not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        import types

        def _shift_right_generic(labels, pad_token_id: int, decoder_start_token_id: int):
            shifted = labels.new_zeros(labels.shape)
            shifted[..., 1:] = labels[..., :-1].clone()
            shifted[..., 0] = decoder_start_token_id
            shifted.masked_fill_(shifted == -100, pad_token_id)
            return shifted

        def _prepare_decoder_input_ids_from_labels(self, labels):
            return _shift_right_generic(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id)

        model.prepare_decoder_input_ids_from_labels = types.MethodType(
            _prepare_decoder_input_ids_from_labels, model)
        info["decoder_input_ids_patch"] = (
            f"{model.__class__.__name__} lacks prepare_decoder_input_ids_from_labels "
            f"-> attached the standard shift-right for label_smoothing_factor>0")

    return tokenizer, model, info


# ==============================================================================
# SECTION 6 — Running one GPU experiment
# ==============================================================================

def configure_torch_threads():
    """Pin torch threads to the CPU affinity + enable TF32 (free on Ampere/A5000)."""
    try:
        torch = _torch()
        n = int(os.environ.get("OMP_NUM_THREADS", available_cpus()))
        torch.set_num_threads(max(1, n))
        # TF32: much faster matrix multiplies on Ampere; the error does not affect
        # seq2seq training at bf16 precision.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = _torch()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _training_args(**kw):
    """Compatibility across transformers versions (evaluation_strategy vs eval_strategy)."""
    import inspect
    from transformers import Seq2SeqTrainingArguments
    params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)
    if "eval_strategy" in params and "evaluation_strategy" not in params:
        if "evaluation_strategy" in kw:
            kw["eval_strategy"] = kw.pop("evaluation_strategy")
    # transformers 5.x renamed two of the arguments used here. Silently dropping
    # them would remove warm-up and group_by_length, i.e. change the protocol.
    # Explicit mapping:
    #   warmup_ratio=r      -> warmup_steps=r  (v5: a float in [0,1) is a ratio)
    #   group_by_length=True-> train_sampling_strategy="group_by_length"
    if "warmup_ratio" in kw and "warmup_ratio" not in params and "warmup_steps" in params:
        kw["warmup_steps"] = float(kw.pop("warmup_ratio"))
    if "group_by_length" in kw and "group_by_length" not in params \
            and "train_sampling_strategy" in params:
        if kw.pop("group_by_length"):
            kw["train_sampling_strategy"] = "group_by_length"
    dropped = sorted(k for k in kw if k not in params)
    # overwrite_output_dir no longer exists in v5 and does not affect Trainer
    # (only HF example scripts use it); any other dropped argument is a protocol error.
    bad = [k for k in dropped if k not in ("overwrite_output_dir",)]
    if bad:
        raise RuntimeError(f"TrainingArguments does not accept {bad} "
                           f"-> the protocol would change; stopping.")
    kw = {k: v for k, v in kw.items() if k in params}
    return Seq2SeqTrainingArguments(**kw)


def _make_trainer(model, args, train_ds, eval_ds, tokenizer, collator, cb):
    import inspect
    from transformers import Seq2SeqTrainer
    params = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
    kw = dict(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
              data_collator=collator, callbacks=cb)
    if "processing_class" in params:
        kw["processing_class"] = tokenizer
    else:
        kw["tokenizer"] = tokenizer

    class _SeqEvalTrainer(Seq2SeqTrainer):
        """transformers 5.x uses LengthGroupedSampler also for eval/predict when
        group_by_length is on, which reorders test predictions relative to
        sources/refs. Eval/predict therefore use a SequentialSampler; the train
        sampler (group_by_length) is unchanged."""

        def _get_eval_sampler(self, eval_dataset):
            from torch.utils.data import SequentialSampler
            if eval_dataset is None:
                return None
            return SequentialSampler(eval_dataset)

    return _SeqEvalTrainer(**kw)


def run_gpu_experiment(job: Dict[str, Any], data_bundle: Dict[str, Any],
                       out_dir: Path, status_cb=None) -> Dict[str, Any]:
    """Wrapper that FREES VRAM whether the job succeeds or fails.

    Each GPU worker runs MANY jobs sequentially in the SAME process (see
    gpu_worker). Without this, a job that fails half-way (OOM, tokenizer or
    generate error, ...) keeps model/optimizer/gradients referenced from the
    crashed stack frame, and the next job on that GPU runs out of memory at
    once. The `finally` below frees VRAM whether the job succeeds or raises.
    """
    torch = _torch()
    handles: Dict[str, Any] = {}
    try:
        return _run_gpu_experiment_body(job, data_bundle, out_dir, status_cb, handles)
    finally:
        import gc
        for k in ("trainer", "model"):
            obj = handles.pop(k, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _run_gpu_experiment_body(job: Dict[str, Any], data_bundle: Dict[str, Any],
                             out_dir: Path, status_cb, handles: Dict[str, Any]
                             ) -> Dict[str, Any]:
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, TrainerCallback

    torch = _torch()
    spec = MODEL_REGISTRY[job["model"]]
    direction = job["direction"]
    strategy = job["strategy"]
    seed = job["seed"]
    scheme = job["split"]
    from_scratch = job["from_scratch"]
    run_id = job["run_id"]

    set_all_seeds(seed)
    configure_torch_threads()
    work_dir = out_dir / "work" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    splits = data_bundle["splits"]
    df_word = data_bundle["df_word"]
    df_train_aug, data_info = build_train_frame(splits["train"], df_word, strategy, seed)

    src_col, tgt_col = (COL_M, COL_V) if direction == "m2v" else (COL_V, COL_M)
    train_texts = (df_train_aug[COL_M].tolist() + df_train_aug[COL_V].tolist())

    if status_cb:
        status_cb(run_id, "loading model")
    tokenizer, model, minfo = load_model_and_tokenizer(
        spec, from_scratch, direction, train_texts, str(work_dir))
    handles["model"] = model   # lets the outer wrapper free VRAM even on errors

    meta = STRATEGY_META[strategy]
    expand_info: Dict[str, Any] = {}
    if meta["expand"] and not (spec.supports_vocab_expand
                               and spec.family != "scratch_transformer"):
        expand_info = {"skipped": f"{spec.key} does not support vocabulary expansion; "
                                  f"strategy {strategy} reduces to 'word'"}
    elif meta["expand"]:
        # Whole-word '▁'+syllable pieces (see expand_vocab_word_boundary);
        # add_tokens-based expansion would split existing words.
        tokenizer, expand_info = expand_vocab_word_boundary(
            tokenizer, model,
            df_word[COL_M].tolist() + df_word[COL_V].tolist(),
            mode=meta["expand"], work_dir=work_dir)

    # ---- language codes for mBART / NLLB ----
    forced_bos = None
    if spec.lang_tokens:
        if spec.family == "nllb":
            mu_c, vi_c = NLLB_MU, NLLB_VI
        else:
            mu_c, vi_c = MBART_MU, MBART_VI
        s_code = mu_c if direction == "m2v" else vi_c
        t_code = vi_c if direction == "m2v" else mu_c
        try:
            tokenizer.src_lang, tokenizer.tgt_lang = s_code, t_code
            forced_bos = tokenizer.convert_tokens_to_ids(t_code)
            if forced_bos is None or forced_bos == tokenizer.unk_token_id:
                raise ValueError(f"language code {t_code} is invalid after adding it")
            minfo["lang_pair"] = f"{s_code}->{t_code}"
        except Exception as e:
            # Safe fallback: use vie_Latn/vi_VN on both sides.
            tokenizer.src_lang = tokenizer.tgt_lang = vi_c
            forced_bos = tokenizer.convert_tokens_to_ids(vi_c)
            minfo["lang_token_fallback"] = f"{type(e).__name__}: {e}"
            minfo["lang_pair"] = f"{vi_c}->{vi_c} (FALLBACK)"

    max_len = spec.max_len
    prefix = PREFIX[direction] if spec.use_prefix else ""

    def preprocess(examples):
        inputs = [prefix + str(x).strip() for x in examples[src_col]]
        targets = [str(x).strip() for x in examples[tgt_col]]
        mi = tokenizer(inputs, max_length=max_len, truncation=True)
        lab = tokenizer(text_target=targets, max_length=max_len, truncation=True)
        mi["labels"] = lab["input_ids"]
        mi["length"] = [len(x) for x in mi["input_ids"]]
        return mi

    ds = {
        "train": Dataset.from_pandas(df_train_aug[[COL_M, COL_V]], preserve_index=False),
        "val": Dataset.from_pandas(splits["val"][[COL_M, COL_V]], preserve_index=False),
        "test": Dataset.from_pandas(splits["test"][[COL_M, COL_V]], preserve_index=False),
    }
    tds = {k: v.map(preprocess, batched=True, remove_columns=[COL_M, COL_V])
           for k, v in ds.items()}

    # SPEED: checkpoint selection needs only a validation subset (paper: 500 of
    # the 1,284 validation pairs). Generating on the full validation set at every
    # epoch is the most expensive part of training. The fixed subset still ranks
    # checkpoints correctly; the FINAL test evaluation is not affected.
    n_val_sel = job.get("val_subset") or 0
    if 0 < n_val_sel < len(tds["val"]):
        tds["val_sel"] = tds["val"].select(range(n_val_sel))
    else:
        tds["val_sel"] = tds["val"]

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    import sacrebleu as sb
    bleu_metric = sb.metrics.BLEU()

    # For vocabulary-expansion runs, validation BLEU is computed against the RAW
    # reference strings (in val_sel / val order) instead of batch_decode(labels),
    # because references passed through an expanded tokenizer and decoded again
    # can hide word-splitting errors. Other runs use batch_decode(labels).
    # Eval/predict use a SequentialSampler (see _make_trainer), so prediction
    # order = dataset order.
    use_raw_val_refs = bool(meta["expand"])
    _raw_val = [str(x).strip() for x in splits["val"][tgt_col].tolist()]
    raw_refs_by_n = {len(tds["val"]): _raw_val,
                     len(tds["val_sel"]): _raw_val[:len(tds["val_sel"])]}
    minfo["val_bleu_refs"] = "raw_strings" if use_raw_val_refs else "decoded_labels"

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        if use_raw_val_refs and len(dp) in raw_refs_by_n:
            dl = raw_refs_by_n[len(dp)]
        else:
            dl = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        dp = [p if p else "∅" for p in dp]
        return {"bleu": bleu_metric.corpus_score(dp, [dl]).score}

    # ---- LR/warm-up for "from scratch" models with a pre-trained HF architecture ----
    # (not used in the paper) spec.lr/spec.optim are tuned for FINE-TUNING. With
    # RANDOM initial weights (from_scratch=True,
    # family != scratch_transformer -- vd byt5-small-scratch, mt5-base-scratch,
    # nllb600-scratch, vit5-*-scratch) that LR is usually TOO HIGH for the first
    # few hundred steps and the loss diverges (BLEU = 0 at every epoch). The
    # scratch_transformer has its own LR/optimizer for random init (lr=3e-4,
    # adamw_torch). Here the LR is divided by 5 and warm-up doubled.
    scratch_pretrained_arch = from_scratch and spec.family != "scratch_transformer"
    eff_lr = spec.lr * 0.2 if scratch_pretrained_arch else spec.lr
    eff_warmup = 0.2 if scratch_pretrained_arch else 0.1
    if scratch_pretrained_arch:
        minfo["scratch_lr_adjusted"] = (
            f"lr {spec.lr} -> {eff_lr}, warmup_ratio -> {eff_warmup}, "
            f"early_stopping_patience -> 6 (avoid divergence and early stops "
            f"when training from random weights)")

    prec = pick_precision()
    ckpt_dir = work_dir / "ckpt"
    args = _training_args(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=False,     # keep checkpoints so runs can resume
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=eff_lr,
        optim=spec.optim,
        lr_scheduler_type="cosine",
        label_smoothing_factor=0.1,
        per_device_train_batch_size=spec.train_bs,
        per_device_eval_batch_size=spec.eval_bs,
        gradient_accumulation_steps=spec.grad_accum,
        gradient_checkpointing=spec.grad_ckpt,
        num_train_epochs=(spec.epochs if not from_scratch
                          else max(spec.epochs + 2, int(spec.epochs * 1.25))),
        warmup_ratio=eff_warmup,
        weight_decay=0.01,
        group_by_length=True,   # batch long with long: 1-3 syllable dictionary pairs
                                # are not padded to the longest sentence in the batch
        length_column_name="length",
        predict_with_generate=True,
        generation_max_length=max_len,
        generation_num_beams=job.get("eval_beams", 1),
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        report_to="none",
        disable_tqdm=True,
        # num_workers=0 ON PURPOSE: gpu_worker() creates the CUDA context (when
        # loading the model) BEFORE Trainer creates its DataLoader. With
        # num_workers>0 the DataLoader fork()s worker processes, and forking a
        # process that holds a CUDA context is unsupported and in practice HANGS
        # forever (GPU 0%, CPU ~0%, no error). The data is already tokenised in
        # memory, so num_workers=0 is hardly slower and removes the deadlock.
        dataloader_num_workers=0,
        **prec,
    )

    class Progress(TrainerCallback):
        def on_epoch_end(self, a, state, control, **kw):
            if status_cb:
                status_cb(run_id,
                          f"epoch {int(state.epoch)}/{int(a.num_train_epochs)}")

        def on_evaluate(self, a, state, control, metrics=None, **kw):
            if status_cb and metrics and "eval_bleu" in metrics:
                status_cb(run_id,
                          f"epoch {int(state.epoch)}/{int(a.num_train_epochs)}"
                          f" | val BLEU {metrics['eval_bleu']:.2f}")

    # ---- patience cho early stopping ----
    # Models trained from random weights need more epochs before they produce
    # a first correct sentence, while validation BLEU stays at 0.0; patience=3
    # (designed for fine-tuning) would stop them too early (not used in the paper).
    patience = 6 if scratch_pretrained_arch else 3
    trainer = _make_trainer(model, args, tds["train"], tds["val_sel"], tokenizer,
                            collator,
                            [EarlyStoppingCallback(early_stopping_patience=patience),
                             Progress()])
    handles["trainer"] = trainer   # lets the outer wrapper free VRAM even on errors
    trainer.compute_metrics = compute_metrics
    if forced_bos is not None:
        # transformers 5 refuses to generate when generation parameters live in
        # model.config ("This strategy to control generation is not supported
        # anymore"), so the target-language code is set on generation_config.
        model.generation_config.forced_bos_token_id = forced_bos

    def _is_oom(exc) -> bool:
        return "out of memory" in str(exc).lower() or \
               exc.__class__.__name__ == "OutOfMemoryError"

    def _is_transient_fs_race(exc) -> bool:
        """OSError [Errno 39] 'Directory not empty' when deleting an old checkpoint.

        Happens on network storage (e.g. NFS volumes on cloud GPU providers)
        because of "silly rename": deleting a file while another process (or
        torch/HF Trainer itself) still holds it open makes NFS create a hidden
        '.nfsXXXX' file, so rmtree() sees a non-empty directory. Not a logic error;
        it clears within seconds once the handle is closed. transformers raises it
        while rotating old checkpoints under save_total_limit.
        """
        return (isinstance(exc, OSError) and getattr(exc, "errno", None) == 39) or \
               "directory not empty" in str(exc).lower()

    # ---- RESUME: continue from an unfinished checkpoint instead of restarting ----
    resume_from = None
    if not job.get("fresh"):
        cks = sorted(ckpt_dir.glob("checkpoint-*"),
                     key=lambda q: int(q.name.split("-")[-1]))
        if cks:
            resume_from = str(cks[-1])
            if status_cb:
                status_cb(run_id, f"resuming from {cks[-1].name}")

    # Gradient checkpointing is OFF by default (max_len 96 keeps memory low) because
    # it costs ~30% extra time. On OOM the configuration falls back step by step:
    # enable checkpointing -> smaller batch -> smaller again, so a run never dies
    # from OOM alone (no run in the paper needed a fallback; see `oom_fallback`).
    # For large models (>= ~1B parameters) most VRAM is optimizer STATE, not
    # activations: AdamW keeps two fp32 moments (8 bytes/parameter), ~10 GB for
    # 1.3B, ~20 GB fixed with fp32 weights + gradients, which a smaller batch cannot
    # fix. The "switch to adafactor" step (factorised second moment, ~0.1 GB)
    # therefore comes BEFORE the batch reductions.
    _opt = spec.optim
    fallbacks = [
        {},
        {"gradient_checkpointing": True},
    ]
    if _opt != "adafactor":
        fallbacks.append({"gradient_checkpointing": True, "optim": "adafactor"})
        _opt = "adafactor"
    fallbacks += [
        {"gradient_checkpointing": True, "optim": _opt,
         "per_device_train_batch_size": max(1, spec.train_bs // 2),
         "gradient_accumulation_steps": spec.grad_accum * 2,
         "per_device_eval_batch_size": max(1, spec.eval_bs // 2)},
        {"gradient_checkpointing": True, "optim": _opt,
         "per_device_train_batch_size": max(1, spec.train_bs // 4),
         "gradient_accumulation_steps": spec.grad_accum * 4,
         "per_device_eval_batch_size": max(1, spec.eval_bs // 4)},
    ]
    t0 = time.time()
    model.config.use_cache = False
    oom_note = None
    attempt = 0
    fs_race_retries = 0
    while attempt < len(fallbacks):
        over = fallbacks[attempt]
        for k, v in over.items():
            setattr(trainer.args, k, v)
        if over:
            oom_note = f"configuration fallback #{attempt}: {over}"
            if status_cb:
                status_cb(run_id, oom_note)
        try:
            trainer.train(resume_from_checkpoint=resume_from)
            break
        except Exception as e:
            if _is_oom(e) and attempt < len(fallbacks) - 1:
                # MUST free the optimizer/scheduler of the failed attempt.
                # Otherwise its state (~10 GB AdamW for a 1.3B model) stays
                # referenced by trainer.optimizer, empty_cache() frees nothing,
                # and the next attempt starts with almost full VRAM. Setting None
                # makes Trainer create a new optimizer for the NEW configuration
                # (Trainer only creates one when self.optimizer is None).
                import gc as _gc
                try:
                    trainer.accelerator.free_memory()
                except Exception:
                    pass
                trainer.optimizer = None
                trainer.lr_scheduler = None
                for p in model.parameters():
                    p.grad = None
                _gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                resume_from = None      # configuration changed -> do not resume
                attempt += 1
                continue
            if _is_transient_fs_race(e) and fs_race_retries < 5:
                # Retry the SAME configuration (this is not an OOM). Reload the
                # NEWEST checkpoint on disk (it may have been written just
                # before the Trainer's checkpoint rotation failed) so no
                # progress is lost.
                fs_race_retries += 1
                wait_s = min(30, 2 ** fs_race_retries)
                msg = (f"transient I/O error on a checkpoint (retry "
                       f"{fs_race_retries}/5 in {wait_s}s): {e}")
                print(f"  {run_id}: {msg}")
                if status_cb:
                    status_cb(run_id, msg)
                time.sleep(wait_s)
                cks = sorted(ckpt_dir.glob("checkpoint-*"),
                             key=lambda q: int(q.name.split("-")[-1]))
                if cks:
                    resume_from = str(cks[-1])
                continue
            if resume_from is None:
                raise
            # corrupt checkpoint (interrupted while writing) -> discard and restart
            print(f"  {run_id}: corrupt checkpoint ({e}); training from scratch again.")
            if status_cb:
                status_cb(run_id, "corrupt checkpoint, restarting training")
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            resume_from = None
            trainer.train()
            break
    model.config.use_cache = True
    train_secs = time.time() - t0
    if oom_note:
        minfo["oom_fallback"] = oom_note

    # ---- save training curves for plotting ----
    curves = []
    for rec in trainer.state.log_history:
        if "epoch" not in rec:
            continue
        row = {"epoch": rec["epoch"]}
        for k in ("loss", "eval_loss", "eval_bleu", "learning_rate", "grad_norm"):
            if k in rec:
                row[k] = rec[k]
        if len(row) > 1:
            curves.append(row)
    (out_dir / "curves").mkdir(parents=True, exist_ok=True)
    (out_dir / "curves" / f"{run_id}.json").write_text(
        json.dumps(curves, ensure_ascii=False), encoding="utf-8")
    best_val_bleu_ = max(
        (c["eval_bleu"] for c in curves if "eval_bleu" in c), default=None)
    epochs_run_ = max((c["epoch"] for c in curves), default=0)


    # ---- CHECKPOINT INTEGRITY CHECK ----
    # transformers warns about "missing keys" for tied embeddings. Usually
    # harmless, but if load_best_model_at_end restored incomplete weights EVERY
    # result would be wrong. Evaluate on validation again and compare with the BEST value of
    # the selection metric itself (eval_bleu), not eval_loss.
    integrity: Dict[str, Any] = {}
    try:
        logged = [r["eval_bleu"] for r in trainer.state.log_history if "eval_bleu" in r]
        if logged:
            trainer.callback_handler.callbacks = [
                c for c in trainer.callback_handler.callbacks
                if not isinstance(c, EarlyStoppingCallback)]
            chk = trainer.evaluate(tds["val"], metric_key_prefix="chk")
            best_logged = max(logged)
            reloaded = float(chk.get("chk_bleu", 0.0))
            integrity = {"metric": "eval_bleu",
                         "best_logged": round(best_logged, 4),
                         "reloaded": round(reloaded, 4),
                         "reloaded_eval_loss": round(float(chk.get("chk_loss", 0.0)), 5)}
            if best_logged > 1.0:      # comparison is meaningless when BLEU ~ 0
                shortfall = (best_logged - reloaded) / best_logged
                integrity["shortfall"] = round(float(shortfall), 5)
                if shortfall > 0.10:
                    integrity["WARNING"] = (
                        "The best checkpoint may NOT have been restored correctly — "
                        "check this run.")
                    print(f"\n  !! {run_id}: val BLEU after reloading "
                          f"({reloaded:.2f}) is {shortfall:.0%} below the best "
                          f"logged value ({best_logged:.2f}).")
            else:
                integrity["note"] = ("Validation BLEU ~ 0 at every epoch, so this "
                                     "cannot be checked; inspect the training run.")
    except Exception as e:
        integrity = {"check_failed": str(e)}

    # ---- Decode the TEST set with beam=4 ----
    if status_cb:
        status_cb(run_id, "decoding test (beam=4)")
    trainer.args.generation_num_beams = 4
    pred = trainer.predict(tds["test"], num_beams=4, max_length=max_len)
    preds = np.where(pred.predictions != -100, pred.predictions, tokenizer.pad_token_id)
    hyps = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]

    sources = splits["test"][src_col].tolist()
    refs = splits["test"][tgt_col].tolist()
    metrics = compute_all_metrics(sources, hyps, refs)

    # ---- latency (batch=1, beam=4) ----
    # Latency depends on hardware + architecture, not on the seed, so it is
    # measured only for the first seed (42), as in the paper (Table II, ms column).
    if job.get("measure_latency", True):
        if status_cb:
            status_cb(run_id, "measuring latency")
        metrics.update(measure_latency(model, tokenizer,
                                       sources[:job.get("latency_n", 100)],
                                       prefix, max_len))

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "run_id": run_id, "experiment": job.get("experiment", ""),
        "model": job["model"], "from_scratch": from_scratch,
        "direction": direction, "strategy": strategy, "split": scheme, "seed": seed,
        "n_params_M": round(n_params / 1e6, 2),
        "n_val_for_selection": len(tds["val_sel"]), "n_val_total": len(tds["val"]),
        "train_seconds": round(train_secs, 1),
        "data_info": data_info, "model_info": minfo, "expand_info": expand_info,
        "ckpt_integrity": integrity,
        "resumed_from": (Path(resume_from).name if resume_from else None),
        "best_val_bleu": best_val_bleu_, "epochs_run": epochs_run_,
        **metrics,
    }

    pd.DataFrame({"source": sources, "reference": refs, "hypothesis": hyps}) \
        .to_csv(out_dir / "predictions" / f"{run_id}.csv", index=False,
                encoding="utf-8-sig")
    # ATOMIC write: an interruption never leaves a corrupt JSON, because
    # resuming relies on this file's existence.
    fin = out_dir / "results" / f"{run_id}.json"
    tmp = fin.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, fin)

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Remove checkpoints: 100+ runs x several GB would fill the disk.
    if not job.get("keep_checkpoints"):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        for junk in ("spm_corpus.txt",):
            (work_dir / junk).unlink(missing_ok=True)
    return result


def measure_latency(model, tokenizer, sources: List[str], prefix: str, max_len: int
                    ) -> Dict[str, float]:
    torch = _torch()
    model.eval()
    dev = next(model.parameters()).device
    times = []
    with torch.no_grad():
        for s in sources[:5]:                                   # warm-up
            enc = tokenizer(prefix + s, return_tensors="pt", truncation=True,
                            max_length=max_len).to(dev)
            model.generate(**enc, num_beams=4, max_length=max_len)
        for s in sources:
            enc = tokenizer(prefix + s, return_tensors="pt", truncation=True,
                            max_length=max_len).to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(**enc, num_beams=4, max_length=max_len)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    a = np.array(times)
    return {"latency_ms_avg": round(float(a.mean()), 3),
            "latency_ms_min": round(float(a.min()), 3),
            "latency_ms_max": round(float(a.max()), 3),
            "latency_n": len(a)}


# ==============================================================================
# SECTION 7 — CPU jobs (separate processes)
# ==============================================================================

def run_cpu_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Baselines that need no GPU: copy/identity + rule-based + corpus statistics."""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"          # one thread per process, no core contention
    try:
        job = payload["job"]
        out_dir = Path(payload["out_dir"])
        df_s, df_w = load_frames(payload["sentence_file"], payload["word_file"])
        df_s, _ = dedup_and_group(df_s, jaccard_thr=payload["jaccard_thr"])
        splits = grouped_split(df_s, SPLIT_SCHEMES[job["split"]], job["seed"])

        direction = job["direction"]
        src_col, tgt_col = (COL_M, COL_V) if direction == "m2v" else (COL_V, COL_M)
        sources = splits["test"][src_col].tolist()
        refs = splits["test"][tgt_col].tolist()

        if job["model"] == "copy":
            hyps = baseline_copy(sources)
            extra = {}
        elif job["model"] == "rule":
            rb = RuleBasedTranslator(df_w, src_col, tgt_col)
            hyps = [rb.translate(s) for s in sources]
            extra = {"lexicon_coverage_pct": round(rb.coverage(sources), 3),
                     "lexicon_entries": len(rb.best)}
        else:
            raise ValueError(job["model"])

        m = compute_all_metrics(sources, hyps, refs)
        res = {"run_id": job["run_id"], "experiment": job.get("experiment", ""),
               "model": job["model"], "from_scratch": False,
               "direction": direction, "strategy": "none", "split": job["split"],
               "seed": job["seed"], "n_params_M": 0.0, "train_seconds": 0.0,
               **extra, **m}
        pd.DataFrame({"source": sources, "reference": refs, "hypothesis": hyps}) \
            .to_csv(out_dir / "predictions" / f"{job['run_id']}.csv",
                    index=False, encoding="utf-8-sig")
        (out_dir / "results" / f"{job['run_id']}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "run_id": job["run_id"], "bleu": res["bleu"]}
    except Exception as e:
        return {"ok": False, "run_id": payload["job"]["run_id"],
                "error": f"{e}\n{traceback.format_exc()}"}


def corpus_analysis(df_s: pd.DataFrame, df_w: pd.DataFrame, out_dir: Path,
                    dedup_stats: Dict[str, int]) -> Dict[str, Any]:
    """Corpus statistics + cognate/tone analysis (Table I)."""
    tm = df_s[COL_M].map(lambda s: len(norm_text(s).split()))
    tv = df_s[COL_V].map(lambda s: len(norm_text(s).split()))
    dens = np.array([cognate_density(m, v) for m, v in zip(df_s[COL_M], df_s[COL_V])])
    sim = np.array([norm_edit_sim(m, v) for m, v in zip(df_s[COL_M], df_s[COL_V])])

    tone_agree = []
    for m, v in zip(df_s[COL_M], df_s[COL_V]):
        a, b = tone_sequence(m), tone_sequence(v)
        n = min(len(a), len(b))
        if n:
            tone_agree.append(sum(a[i] == b[i] for i in range(n)) / n)

    vocab_m = set(itertools.chain.from_iterable(norm_text(s).split() for s in df_s[COL_M]))
    vocab_v = set(itertools.chain.from_iterable(norm_text(s).split() for s in df_s[COL_V]))

    res = {
        "dedup": dedup_stats,
        "n_sentence_pairs": len(df_s),
        "n_word_pairs": len(df_w),
        "avg_tokens_muong": round(float(tm.mean()), 3),
        "avg_tokens_viet": round(float(tv.mean()), 3),
        "length_ratio_M_over_V": round(float(tm.sum() / max(tv.sum(), 1)), 4),
        "vocab_size_muong": len(vocab_m),
        "vocab_size_viet": len(vocab_v),
        "vocab_overlap_pct": round(100 * len(vocab_m & vocab_v) / max(len(vocab_m | vocab_v), 1), 3),
        "cognate_density_mean": round(float(dens.mean()), 4),
        "cognate_density_quartiles": [round(float(x), 4)
                                      for x in np.quantile(dens, [.25, .5, .75])],
        "sentence_char_similarity_mean": round(float(sim.mean()), 4),
        "tone_agreement_mean": round(float(np.mean(tone_agree)) if tone_agree else 0.0, 4),
        "identical_after_tone_strip_pct": round(100 * float(np.mean(
            [strip_tones(norm_text(m)) == strip_tones(norm_text(v))
             for m, v in zip(df_s[COL_M], df_s[COL_V])])), 3),
        "note": "Quantitative corpus statistics (Table I): surface overlap, cognate "
                "density, character similarity and tone agreement.",
    }

    def _hist(a, bins, rng=None):
        c, e = np.histogram(np.asarray(a, dtype=float), bins=bins, range=rng)
        return {"counts": c.tolist(), "edges": e.tolist()}

    hi = int(max(tm.max(), tv.max()))
    res["histograms"] = {
        "len_muong": _hist(tm.values, bins=min(40, max(5, hi)), rng=(0, hi)),
        "len_viet": _hist(tv.values, bins=min(40, max(5, hi)), rng=(0, hi)),
        "cognate": _hist(dens, bins=25, rng=(0.0, 1.0)),
        "charsim": _hist(sim, bins=25, rng=(0.0, 1.0)),
        "tone_agreement": _hist(tone_agree or [0.0], bins=25, rng=(0.0, 1.0)),
    }
    (out_dir / "corpus_analysis.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res


# ==============================================================================
# SECTION 8 — Job planning
# ==============================================================================

def make_run_id(model: str, scratch: bool, direction: str, strategy: str,
                split: str, seed: int) -> str:
    tag = f"{model}{'-scratch' if scratch else ''}"
    return f"{tag}__{direction}__{strategy}__{split.replace(':', '-')}__s{seed}"


def _variance_seeds_list(args) -> List[int]:
    """Seeds for E5: continue from E1's first seed so both experiments share cells."""
    base = args.seeds[0] if args.seeds else 42
    return [base + i for i in range(args.variance_seeds)]


def plan_jobs(args) -> Tuple[List[Dict], List[Dict]]:
    """Create jobs PER EXPERIMENT; every experiment is a full grid."""
    cpu_jobs: List[Dict] = []
    gpu_jobs: List[Dict] = []
    dirs = args.directions
    main = args.main_split
    wanted = set(args.experiments)

    def gpu(exp, model, scratch, direction, strategy, split, seed):
        gpu_jobs.append(dict(
            run_id=make_run_id(model, scratch, direction, strategy, split, seed),
            experiment=exp, model=model, direction=direction, strategy=strategy,
            split=split, seed=seed, from_scratch=scratch, kind="gpu"))

    # --- E0: CPU baselines, every split, direction and seed (almost free) ---
    all_seeds = sorted(set(args.seeds) | set(_variance_seeds_list(args)))
    for mdl, d, sp, sd in itertools.product(["copy", "rule"], dirs,
                                            args.splits, all_seeds):
        cpu_jobs.append(dict(run_id=make_run_id(mdl, False, d, "none", sp, sd),
                             experiment="E0_baseline", model=mdl, direction=d,
                             strategy="none", split=sp, seed=sd,
                             from_scratch=False, kind="cpu"))

    # --- E1: main table. Full grid, same seeds for EVERY model. ---
    if "E1_main" in wanted:
        for mdl, st, d, sd in itertools.product(args.models, ["base", "word"],
                                                dirs, args.seeds):
            gpu("E1_main", mdl, False, d, st, main, sd)

    # --- E5: seed variance on ONE model with many seeds ---
    # When E1 already has >= 3 seeds it gives mean±std for EVERY model, so E5
    # would be redundant (two places reporting variance with different seeds).
    if "E5_seed_variance" in wanted and len(args.seeds) >= 3:
        print("  [plan] E1 already has >= 3 seeds for every model => E5 skipped "
              "(mean±std comes directly from E1).")
    elif "E5_seed_variance" in wanted:
        for st, d, sd in itertools.product(["base", "word"], dirs,
                                           _variance_seeds_list(args)):
            gpu("E5_seed_variance", args.variance_model, False, d, st, main, sd)

    # --- E2: from scratch (not in the paper). Full grid on the declared models. ---
    if "E2_scratch" in wanted:
        for mdl, st, d, sd in itertools.product(args.scratch_models,
                                                ["base", "word"], dirs,
                                                args.seeds[:args.ablation_seeds]):
            gpu("E2_scratch", mdl, True, d, st, main, sd)
        if args.include_transformer_scratch:
            for st, d, sd in itertools.product(["base", "word"], dirs,
                                               args.seeds[:args.ablation_seeds]):
                gpu("E2_scratch", "transformer-small", True, d, st, main, sd)

    # --- E3: every data strategy on ONE model (ViT5-base in the paper) ---
    if "E3_strategy" in wanted:
        for st, d, sd in itertools.product(STRATEGIES, dirs,
                                           args.seeds[:args.ablation_seeds]):
            gpu("E3_strategy", args.ablation_model, False, d, st, main, sd)

    # --- E4: data contraction. Cells at the main split are reused from E1/E3. --
    if "E4_contraction" in wanted:
        for st, d, sp, sd in itertools.product(
                ["base", "word"], dirs,
                [x for x in args.splits if x != main],
                args.seeds[:args.ablation_seeds]):
            gpu("E4_contraction", args.ablation_model, False, d, st, sp, sd)

    def uniq(js):
        seen, out = set(), []
        for j in js:
            if j["run_id"] not in seen:
                seen.add(j["run_id"])
                out.append(j)
        return out

    return uniq(cpu_jobs), uniq(gpu_jobs)


def apply_budget(gpu_jobs: List[Dict], costs: Dict[str, float], budget_hours: float,
                 sec_per_cost: float, order: List[str]
                 ) -> Tuple[List[Dict], List[Dict], List[str]]:
    """Keep WHOLE highest-priority experiments that fit the budget.

    Never cut cells inside an experiment: a table with holes (one model with
    three seeds, another without) is worse than no table.
    """
    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for j in gpu_jobs:
        by_exp[j["experiment"]].append(j)

    keep: List[Dict] = []
    drop: List[Dict] = []
    dropped_exps: List[str] = []
    used = 0.0
    for exp in order:
        js = by_exp.get(exp)
        if not js:
            continue
        h = sum(costs[j["run_id"]] for j in js) * sec_per_cost / 3600
        if used + h <= budget_hours:
            keep.extend(js)
            used += h
        else:
            drop.extend(js)
            dropped_exps.append(exp)
    for exp, js in by_exp.items():          # experiments not listed in `order`
        if exp not in order:
            keep.extend(js)
    return keep, drop, dropped_exps


def estimate_cost(job: Dict, n_train: int) -> float:
    """Relative cost unit used for the ETA."""
    spec = MODEL_REGISTRY.get(job["model"])
    if spec is None:
        return 0.01
    base = spec.cost * spec.epochs * (n_train / 10000.0)
    if job["from_scratch"]:
        base *= 1.25
    frac = STRATEGY_META.get(job["strategy"], {}).get("word", 0)
    if frac > 0:
        # +15k dictionary pairs, but each has only 1-3 syllables and group_by_length
        # batches them together, so the real cost is much lower than the count.
        base *= 1.0 + 0.25 * frac
    return base


# ==============================================================================
# SECTION 9 — Dashboard
# ==============================================================================

def fmt_hms(sec: float) -> str:
    if sec is None or not math.isfinite(sec) or sec < 0:
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


class Dashboard:
    """Live progress table, one row per GPU (rich if available, plain text otherwise)."""

    def __init__(self, gpu_jobs, cpu_jobs, costs, gpus: List[str], sec_prior: float):
        self.gpu_jobs = gpu_jobs
        self.cpu_jobs = cpu_jobs
        self.costs = costs
        self.gpus = list(gpus)
        self.n_gpu = max(len(self.gpus), 1)
        self.total_cost = sum(costs.get(j["run_id"], 0.0) for j in gpu_jobs) or 1.0
        self.done_cost = 0.0
        self.sec_per_cost: Optional[float] = sec_prior
        self.t0 = time.time()
        self.cur: Dict[str, Optional[str]] = {g: None for g in self.gpus}
        self.msg: Dict[str, str] = {g: "waiting" for g in self.gpus}
        self.finished: List[Tuple[str, Optional[float], float]] = []
        self.cpu_done = 0
        self.cpu_total = len(cpu_jobs)
        self.errors: List[str] = []
        try:
            from rich.console import Console
            from rich.live import Live
            self._rich = True
            self.console = Console()
            self.Live = Live
        except Exception:
            self._rich = False
            self.console = None
            self.Live = None
        self._live = None

    # -------- ETA --------
    def eta(self) -> Optional[float]:
        if not self.gpu_jobs:
            return 0.0
        if self.sec_per_cost is None:
            return None
        # divide by the number of GPUs: jobs run in parallel
        return max(self.total_cost - self.done_cost, 0.0) * self.sec_per_cost / self.n_gpu

    def note_finished(self, run_id: str, bleu: Optional[float], secs: float):
        c = self.costs.get(run_id, 0.0)
        self.done_cost += c
        if c > 0 and bleu is not None:
            obs = secs / c
            self.sec_per_cost = obs if self.sec_per_cost is None \
                else 0.7 * self.sec_per_cost + 0.3 * obs
        self.finished.append((run_id, bleu, secs))

    # -------- render --------
    def _table(self):
        from rich.panel import Panel
        from rich.table import Table
        from rich.console import Group

        head = Table.grid(padding=(0, 2))
        head.add_row("[bold]Run GPU[/]", f"{len(self.finished)}/{len(self.gpu_jobs)}"
                                         f"   ({self.n_gpu} GPU song song)")
        head.add_row("[bold]Run CPU[/]", f"{self.cpu_done}/{self.cpu_total}")
        head.add_row("[bold]Elapsed[/]", fmt_hms(time.time() - self.t0))
        e = self.eta()
        head.add_row("[bold]ETA[/]",
                     f"{fmt_hms(e)}  (~{e/86400:.1f} days)" if e else "calibrating...")
        pct = (100.0 * self.done_cost / self.total_cost) if self.gpu_jobs \
            else (100.0 * self.cpu_done / max(self.cpu_total, 1))
        n = max(0, min(40, int(pct / 2.5)))
        head.add_row("[bold]Progress[/]", f"{'█' * n}{'░' * (40 - n)} {pct:5.1f}%")

        g = Table(show_header=True, header_style="bold cyan", expand=True)
        g.add_column("GPU", width=5)
        g.add_column("Running", overflow="fold")
        g.add_column("Status", overflow="fold")
        for gid in self.gpus:
            g.add_row(str(gid), self.cur.get(gid) or "-", self.msg.get(gid, "-"))

        t = Table(show_header=True, header_style="bold cyan", expand=True)
        t.add_column("Recent runs", overflow="fold")
        t.add_column("BLEU", justify="right", width=8)
        t.add_column("Time", justify="right", width=10)
        for rid, bl, sc in self.finished[-8:]:
            t.add_row(rid, f"{bl:.2f}" if bl is not None else "ERROR", fmt_hms(sc))

        parts = [head, g, t]
        if self.errors:
            et = Table(show_header=True, header_style="bold red", expand=True)
            et.add_column("Errors (last 5)", overflow="fold")
            for x in self.errors[-5:]:
                et.add_row(x)
            parts.append(et)
        return Panel(Group(*parts), title="VietMuong — Experiment Runner",
                     border_style="green")

    def start(self):
        if self._rich:
            self._live = self.Live(self._table(), refresh_per_second=4,
                                   console=self.console)
            self._live.start()

    def refresh(self):
        if self._rich and self._live:
            self._live.update(self._table())
        else:
            pct = 100.0 * self.done_cost / self.total_cost if self.gpu_jobs else 0
            cur = " | ".join(f"gpu{g}:{(self.cur.get(g) or '-')[:36]}" for g in self.gpus)
            print(f"[{pct:5.1f}%] {len(self.finished)}/{len(self.gpu_jobs)} GPU, "
                  f"{self.cpu_done}/{self.cpu_total} CPU | {cur} | "
                  f"ETA {fmt_hms(self.eta())}", flush=True)

    def stop(self):
        if self._rich and self._live:
            self._live.stop()


# ==============================================================================
# SECTION 9b — GPU worker (one process per GPU)
# ==============================================================================

def gpu_worker(gpu_id: str, job_q, status_q, cfg: Dict[str, Any]) -> None:
    """Runs in its own process, pinned to EXACTLY ONE GPU.

    CUDA_VISIBLE_DEVICES must be set BEFORE torch is imported. The process is
    created with 'spawn', so it re-imports this module, which does not import
    torch at top level — pinning the GPU here is therefore safe.
    """
    import queue as _queue
    # gpu_id is "<device>" or "<device>#<k>" when several workers share one
    # physical GPU. The part after '#' is only a dashboard label, NOT a device
    # index.
    _device = str(gpu_id).split("#")[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = _device
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Reduce VRAM fragmentation between consecutive jobs in one process
    # (as suggested by the CUDA OOM message itself).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.get("epochs_override"):
        for _sp in MODEL_REGISTRY.values():
            _sp.epochs = int(cfg["epochs_override"])
    if cfg.get("max_len_override"):
        for _sp in MODEL_REGISTRY.values():
            _sp.max_len = int(cfg["max_len_override"])
    n_thread = cfg.get("torch_threads") or max(1, available_cpus() // max(cfg["n_gpu"], 1))
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = str(n_thread)

    # Several workers on one GPU: give each process a VRAM cap. Without it the
    # first worker to load a model takes almost all VRAM and the other OOMs
    # immediately. With a cap, a worker that exceeds it OOMs EARLY and falls
    # back through the configuration ladder (gradient checkpointing ->
    # adafactor -> smaller batch). The paper's runs used one worker per GPU.
    _wpg = int(cfg.get("workers_per_gpu", 1) or 1)
    if _wpg > 1:
        try:
            _t = _torch()
            if _t.cuda.is_available():
                _t.cuda.set_per_process_memory_fraction(0.96 / _wpg, 0)
        except Exception:
            pass

    out_dir = Path(cfg["out_dir"])
    try:
        df_s, df_w = load_frames(cfg["sentence_file"], cfg["word_file"])
        df_s, _ = dedup_and_group(df_s, jaccard_thr=cfg["jaccard_thr"])
    except Exception as e:
        status_q.put({"type": "fatal", "gpu": gpu_id, "error": str(e)})
        return

    split_cache: Dict[Tuple[str, int], Dict[str, pd.DataFrame]] = {}
    status_q.put({"type": "ready", "gpu": gpu_id})

    def cb(run_id, msg):
        status_q.put({"type": "status", "gpu": gpu_id, "run_id": run_id, "msg": msg})

    while True:
        try:
            job = job_q.get_nowait()
        except _queue.Empty:
            break
        except Exception:
            break
        if job is None:
            break

        run_id = job["run_id"]
        status_q.put({"type": "start", "gpu": gpu_id, "run_id": run_id})
        t0 = time.time()
        try:
            key = (job["split"], job["seed"])
            if key not in split_cache:
                split_cache[key] = grouped_split(
                    df_s, SPLIT_SCHEMES[job["split"]], job["seed"])
            splits = split_cache[key]

            dfw = df_w
            if cfg["exclude_test_words"]:
                dfw, removed = filter_word_pairs_leaking_into_test(df_w, splits["test"])
                cb(run_id, f"removed {removed} dictionary pairs leaking into test")

            res = run_gpu_experiment(job, {"splits": splits, "df_word": dfw},
                                     out_dir, status_cb=cb)
            status_q.put({"type": "done", "gpu": gpu_id, "run_id": run_id,
                          "bleu": res.get("bleu"), "secs": time.time() - t0})
        except Exception as e:
            (out_dir / "logs" / f"{run_id}.err").write_text(
                traceback.format_exc(), encoding="utf-8")
            status_q.put({"type": "done", "gpu": gpu_id, "run_id": run_id,
                          "bleu": None, "secs": time.time() - t0,
                          "error": f"{type(e).__name__}: {e}"})
    status_q.put({"type": "exit", "gpu": gpu_id})


# ==============================================================================
# SECTION 10 — Report aggregation
# ==============================================================================

def aggregate(out_dir: Path) -> Optional[pd.DataFrame]:
    files = sorted((out_dir / "results").glob("*.json"))
    if not files:
        print("No results to aggregate yet.")
        return None
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            d.pop("tone_confusions_top10", None)
            for k in ("data_info", "model_info", "expand_info", "ckpt_integrity"):
                v = d.pop(k, None)
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float, str, bool)):
                            d[f"{k}.{kk}"] = vv
            rows.append(d)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_raw.csv", index=False, encoding="utf-8-sig")

    keys = ["model", "from_scratch", "direction", "strategy", "split"]
    metric_cols = [c for c in ["bleu", "chrf2", "ter", "tone_accuracy",
                               "copy_rate_exact", "len_ratio_hyp_ref",
                               "latency_ms_avg", "n_params_M"] if c in df.columns]
    agg = df.groupby(keys)[metric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "summary_mean_std.csv", index=False, encoding="utf-8-sig")

    # ---- main LaTeX table ----
    try:
        top_split = df["split"].value_counts().idxmax()
        main = agg[agg["split"] == top_split].copy()
        main["BLEU"] = main.apply(
            lambda r: f"{r['bleu_mean']:.2f}$\\pm${0 if pd.isna(r['bleu_std']) else r['bleu_std']:.2f}",
            axis=1)
        main["chrF++"] = main.apply(
            lambda r: f"{r['chrf2_mean']:.2f}", axis=1)
        piv = main.pivot_table(index=["model", "from_scratch", "strategy"],
                               columns="direction", values="BLEU", aggfunc="first")
        (out_dir / "table_main.tex").write_text(
            piv.to_latex(escape=False, na_rep="--"), encoding="utf-8")
    except Exception as e:
        print(f"Could not write the LaTeX table: {e}")

    # ---- significance tests (paired bootstrap, 1,000 resamples) ----
    # (a) every system vs the COPY baseline -> the system translates, not copies
    # (b) word/token/... strategies vs 'base' of the SAME model -> is the gain real
    sig_rows: List[Dict[str, Any]] = []
    pred_dir = out_dir / "predictions"

    def _pb(run_a: str, run_b: str, label: str):
        fa, fb = pred_dir / f"{run_a}.csv", pred_dir / f"{run_b}.csv"
        if run_a == run_b or not (fa.exists() and fb.exists()):
            return
        try:
            A = pd.read_csv(fa).fillna("")
            B = pd.read_csv(fb).fillna("")
            if len(A) != len(B) or len(A) == 0:
                return
            st = paired_bootstrap(A["hypothesis"].astype(str).tolist(),
                                  B["hypothesis"].astype(str).tolist(),
                                  A["reference"].astype(str).tolist(),
                                  n_samples=1000)
            sig_rows.append({"run_id": run_a, "baseline": run_b, "comparison": label,
                             "significant_p05": st["p_value"] < 0.05, **st})
        except Exception as e:
            print(f"  bootstrap {run_a} vs {run_b}: {e}")

    for _, r in df.iterrows():
        if r["model"] in ("copy",):
            continue
        _pb(r["run_id"],
            make_run_id("copy", False, r["direction"], "none", r["split"], r["seed"]),
            "vs_copy")
        if r["strategy"] not in ("base", "none"):
            _pb(r["run_id"],
                make_run_id(r["model"], bool(r["from_scratch"]), r["direction"],
                            "base", r["split"], r["seed"]),
                "vs_same_model_base")
        # The paper compares token/token_avg with word mixing 100% (same 25,649
        # instances) and the two initialisations with each other.
        if r["strategy"] in ("token", "token_avg", "word_token"):
            _pb(r["run_id"],
                make_run_id(r["model"], bool(r["from_scratch"]), r["direction"],
                            "word", r["split"], r["seed"]),
                "vs_same_model_word")
        if r["strategy"] == "token":
            _pb(r["run_id"],
                make_run_id(r["model"], bool(r["from_scratch"]), r["direction"],
                            "token_avg", r["split"], r["seed"]),
                "token_vs_token_avg")

    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv",
                                      index=False, encoding="utf-8-sig")

    print(f"\nWritten:\n  {out_dir/'summary_raw.csv'}\n  "
          f"{out_dir/'summary_mean_std.csv'}\n  {out_dir/'table_main.tex'}")
    if sig_rows:
        print(f"  {out_dir/'significance.csv'}  ({len(sig_rows)} comparisons)")
    return agg


# ==============================================================================
# SECTION 10b — PLOTS
# ==============================================================================
# Validated palette (CVD-safe, OKLab ΔE ≥ 8 between adjacent pairs, light
# background). TWO RULES:
#   1. Colours are assigned in a FIXED order and only to entities that are
#      actually drawn — the palette is never cycled (a 9th series is merged,
#      it never reuses slot 1).
#   2. Colour follows the entity, not its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDINAL3 = ["#86b6ef", "#2a78d6", "#104281"]        # single-hue blue ramp, light→dark
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2de"
MAX_SERIES = len(SERIES)

STRATEGY_LABEL = {
    "base": "Base (sentences only)", "word": "Word-mixed",
    "token": "Token expansion (random init)", "token_avg": "Token expansion (avg init)",
    "word_token": "Word-mixed + Token expansion",
    "ctrl_shuffle": "CTRL: shuffled word pairs", "ctrl_dup": "CTRL: duplicated sentences",
    "word_p25": "Word-mixed 25%", "word_p50": "Word-mixed 50%",
    "word_p75": "Word-mixed 75%", "none": "Baseline",
}
DIR_LABEL = {"m2v": "Muong → Vietnamese", "v2m": "Vietnamese → Muong"}
# Short tick labels — the full labels above are for legends.
STRATEGY_SHORT = {
    "base": "Base", "word": "Word-mixed", "token": "Tok (rand)",
    "token_avg": "Tok (avg)", "word_token": "Word+Tok",
    "ctrl_shuffle": "CTRL-shuffled", "ctrl_dup": "CTRL-duplicated",
    "word_p25": "Word 25%", "word_p50": "Word 50%", "word_p75": "Word 75%",
    "none": "Baseline",
}
HEADLINE = ["base", "word", "none"]     # strategies shown in the overview figures


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "figure.dpi": 120, "savefig.bbox": "tight",
        "axes.unicode_minus": False,
    })
    return plt


def assign_colors(entities: Sequence[Any]) -> Tuple[Dict[Any, str], Dict[Any, str], List[Any]]:
    """Assign colour + marker in a fixed order to exactly the entities that are drawn.

    More than 8 entities are TRUNCATED (with a warning) instead of cycling the
    palette, which would give two different lines the same colour.
    """
    ents = list(dict.fromkeys(entities))
    if len(ents) > MAX_SERIES:
        print(f"  [figure] {len(ents)} series exceed the limit of {MAX_SERIES} safe colours "
              f"— only the first {MAX_SERIES} are drawn; split the rest into another figure.")
        ents = ents[:MAX_SERIES]
    cmap = {e: SERIES[i] for i, e in enumerate(ents)}
    mmap = {e: MARKERS[i] for i, e in enumerate(ents)}
    return cmap, mmap, ents


def _clean(ax, ygrid=True, xgrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", linestyle="-", linewidth=0.7, color=GRID)
    if xgrid:
        ax.grid(axis="x", linestyle="-", linewidth=0.7, color=GRID)
    return ax


def _save(fig, out_dir: Path, name: str):
    d = out_dir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")        # vector, for LaTeX
    fig.savefig(d / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _note(ax, x, y, text, ha="right", va="bottom", dy=4):
    """Label with a background in the plot colour so it does not cover marks."""
    ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=7.5, color=INK2,
                bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.9))


def _bar_labels(ax, bars, fmt="{:.1f}", fontsize=7, horizontal=False):
    """Value labels directly on the bars.

    Required: some palette colours are below 3:1 contrast against the background,
    so values must be visible as labels or in an accompanying table.
    """
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if v is None or not math.isfinite(v) or v == 0:
            continue
        if horizontal:
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2),
                        textcoords="offset points", xytext=(3, 0),
                        ha="left", va="center", fontsize=fontsize, color=INK2)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=fontsize, color=INK2)


def _tag(df: pd.DataFrame) -> pd.Series:
    return df["model"] + df["from_scratch"].map({True: "-scratch", False: ""})


def _order_models(d: pd.DataFrame) -> List[str]:
    """Pre-trained models first (descending BLEU), then from-scratch variants."""
    m = d.groupby("tag")["bleu"].mean()
    pre = sorted([t for t in m.index if "-scratch" not in t],
                 key=lambda t: -m[t])
    scr = sorted([t for t in m.index if "-scratch" in t], key=lambda t: -m[t])
    return pre + scr


# ------------------------------------------------------------------ curves ---
def _curve_frame(out_dir: Path, df: pd.DataFrame) -> pd.DataFrame:
    cdir = out_dir / "curves"
    if not cdir.exists():
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        f = cdir / f"{r['run_id']}.json"
        if not f.exists():
            continue
        try:
            cur = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        tag = f"{r['model']}{'-scratch' if r.get('from_scratch') else ''}"
        for c in cur:
            rows.append({"model": tag, "strategy": r["strategy"],
                         "direction": r["direction"], "seed": r["seed"], **c})
    return pd.DataFrame(rows)


def _plot_curve_panels(out_dir: Path, cdf: pd.DataFrame, metric: str, ylabel: str,
                       fname: str, title: str):
    if metric not in cdf.columns:
        return
    sub = cdf.dropna(subset=[metric])
    if len(sub) == 0:
        return
    plt = _mpl()
    models = sorted(sub["model"].unique())
    ncol = min(3, len(models))
    nrow = math.ceil(len(models) / ncol)
    strategies = sorted(sub["strategy"].unique(),
                        key=lambda s: STRATEGIES.index(s) if s in STRATEGIES else 99)
    cmap, mmap, strategies = assign_colors(strategies)

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(max(6.6, 4.0 * ncol), 3.4 * nrow),
                             squeeze=False)
    for k, mdl in enumerate(models):
        ax = _clean(axes[k // ncol][k % ncol])
        for st in strategies:
            for d in sorted(sub["direction"].unique()):
                g = sub[(sub["model"] == mdl) & (sub["strategy"] == st)
                        & (sub["direction"] == d)]
                if len(g) == 0:
                    continue
                m = g.groupby("epoch")[metric].mean().sort_index()
                ax.plot(m.index, m.values, color=cmap[st], linewidth=2.0,
                        marker=mmap[st], markersize=4.5,
                        markeredgecolor=SURFACE, markeredgewidth=0.8,
                        linestyle="-" if d == "m2v" else "--", alpha=0.95)
        ax.set_title(mdl, color=INK, loc="left")
        ax.set_xlabel("epoch")
        if k % ncol == 0:
            ax.set_ylabel(ylabel)
    for k in range(len(models), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    handles = [plt.Line2D([], [], color=cmap[s], lw=2.2, marker=mmap[s],
                          markersize=5, markeredgecolor=SURFACE,
                          label=STRATEGY_LABEL.get(s, s)) for s in strategies]
    handles += [plt.Line2D([], [], color=INK2, lw=2, ls="-", label=DIR_LABEL["m2v"]),
                plt.Line2D([], [], color=INK2, lw=2, ls="--", label=DIR_LABEL["v2m"])]
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)),
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, out_dir, fname)


def plot_training_curves(out_dir: Path, df: pd.DataFrame):
    """Small multiples: one panel per model, one line per strategy."""
    cdf = _curve_frame(out_dir, df)
    if len(cdf) == 0:
        return
    metrics = [("eval_bleu", "Validation BLEU", "val_bleu",
                "Validation BLEU per epoch"),
               ("eval_loss", "Validation loss", "eval_loss",
                "Validation loss per epoch"),
               ("loss", "Training loss", "train_loss",
                "Training loss per epoch")]

    # (a) Overview: base vs word only => exactly two series.
    head = cdf[cdf["strategy"].isin(HEADLINE)]
    for met, ylab, fn, title in metrics:
        _plot_curve_panels(out_dir, head, met, ylab,
                           f"fig_training_curve_{fn}",
                           f"{title} — Base vs. Word-mixed")

    # (b) Ablation: models with >2 strategies get their own figure.
    cnt = cdf.groupby("model")["strategy"].nunique()
    for mdl in cnt[cnt > 2].index:
        sub = cdf[(cdf["model"] == mdl)
                  & (~cdf["strategy"].isin(["word_p25", "word_p50", "word_p75"]))]
        if sub["strategy"].nunique() < 3:
            continue
        _plot_curve_panels(out_dir, sub, "eval_bleu", "Validation BLEU",
                           f"fig_training_curve_ablation_{mdl}",
                           f"Validation BLEU per epoch — data-strategy ablation "
                           f"on {mdl}")


# --------------------------------------------------------------- grouped bars ---
def _grouped_bars(out_dir: Path, d: pd.DataFrame, xcol: str, groupcol: str,
                  x_order: List[str], group_order: List[str],
                  group_label, xlabel_map, copy_ref: Dict[str, float],
                  title: str, fname: str, ylab: str = "BLEU"):
    plt = _mpl()
    g = d.groupby(["direction", xcol, groupcol])["bleu"] \
         .agg(["mean", "std"]).reset_index()
    if len(g) == 0:
        return
    xs_lab = [x for x in x_order if x in set(g[xcol])]
    groups = [s for s in group_order if s in set(g[groupcol])]
    if not xs_lab or not groups:
        return
    cmap, _, groups = assign_colors(groups)
    dirs = sorted(g["direction"].unique())

    width_in = min(13.5, max(6.5, 0.62 * len(xs_lab) * max(len(groups), 2)))
    fig, axes = plt.subplots(len(dirs), 1, figsize=(width_in, 3.5 * len(dirs)),
                             squeeze=False, sharex=True)
    w = 0.82 / len(groups)
    for i, dr in enumerate(dirs):
        ax = _clean(axes[i][0])
        for j, st in enumerate(groups):
            sub = g[(g["direction"] == dr) & (g[groupcol] == st)]
            sub = sub[sub[xcol].isin(xs_lab)]
            if len(sub) == 0:
                continue
            xs = [xs_lab.index(t) + (j - (len(groups) - 1) / 2) * w for t in sub[xcol]]
            bars = ax.bar(xs, sub["mean"], width=w * 0.88, color=cmap[st],
                          edgecolor=SURFACE, linewidth=1.6, label=group_label(st))
            err = sub["std"].fillna(0).values
            if err.any():
                ax.errorbar(xs, sub["mean"], yerr=err, fmt="none",
                            ecolor=INK2, elinewidth=1.0, capsize=2.5)
            _bar_labels(ax, bars)
        if dr in copy_ref:
            ax.axhline(copy_ref[dr], color=INK2, ls=":", lw=1.4)
            _note(ax, len(xs_lab) - 0.45, copy_ref[dr],
                  f"COPY baseline = {copy_ref[dr]:.1f}", va="top", dy=-4)
        ax.set_ylabel(ylab)
        ax.set_title(DIR_LABEL.get(dr, dr), loc="left", color=INK)
        ax.margins(x=0.02)
        if i == len(dirs) - 1:
            ax.set_xticks(range(len(xs_lab)))
            ax.set_xticklabels([xlabel_map(t) for t in xs_lab],
                               rotation=20, ha="right", fontsize=8)
        if i == 0 and len(groups) >= 2:
            ax.legend(ncol=min(4, len(groups)), loc="upper left",
                      bbox_to_anchor=(0, 1.30))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, fname)


def plot_main_comparison(out_dir: Path, df: pd.DataFrame, main_split: str):
    d = df[df["split"] == main_split].copy()
    if len(d) == 0:
        return
    d["tag"] = _tag(d)
    copy_ref = d[d["model"] == "copy"].groupby("direction")["bleu"].mean().to_dict()
    d = d[~d["model"].isin(["copy", "rule"])]
    if len(d) == 0:
        return

    # (a) MAIN TABLE: models x {base, word}
    head = d[d["strategy"].isin(["base", "word"])]
    if len(head):
        _grouped_bars(out_dir, head, "tag", "strategy", _order_models(head),
                      ["base", "word"], lambda s: STRATEGY_LABEL.get(s, s),
                      lambda t: t, copy_ref,
                      f"Test-set BLEU ({main_split} split) — mean ± s.d. over "
                      f"random seeds", "fig_main_results_bleu")

    # (b) Strategy ABLATION: separate figure so the plot does not grow without bound
    cnt = d.groupby("tag")["strategy"].nunique()
    for tag in cnt[cnt > 2].index:
        sub = d[d["tag"] == tag]
        order = [s for s in STRATEGIES if s in set(sub["strategy"])]
        _grouped_bars(out_dir, sub, "strategy", "tag", order, [tag],
                      lambda s: s, lambda s: STRATEGY_SHORT.get(s, s), copy_ref,
                      f"Data-strategy ablation — {tag} ({main_split} split)",
                      f"fig_strategy_ablation_{tag.replace('/', '-')}")


# ----------------------------------------------------------- scaling curve ---
def plot_scaling_curve(out_dir: Path, df: pd.DataFrame, main_split: str):
    plt = _mpl()
    pct = {"base": 0, "word_p25": 25, "word_p50": 50, "word_p75": 75, "word": 100}
    d = df[(df["split"] == main_split) & (df["strategy"].isin(pct))].copy()
    if d["strategy"].nunique() < 3:
        return
    d["pct"] = d["strategy"].map(pct)
    d["tag"] = _tag(d)

    # Keep only lines that REALLY have enough points, THEN assign colours.
    series = {}
    for t in sorted(d["tag"].unique()):
        for dr in sorted(d["direction"].unique()):
            g = d[(d["tag"] == t) & (d["direction"] == dr)] \
                .groupby("pct")["bleu"].agg(["mean", "std"]).sort_index()
            if len(g) >= 3:
                series[(t, dr)] = g
    if not series:
        return
    cmap, mmap, keys = assign_colors(list(series))

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    _clean(ax)
    for k in keys:
        g, c = series[k], cmap[k]
        ax.plot(g.index, g["mean"], color=c, lw=2.0, marker=mmap[k], markersize=6.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0,
                label=f"{k[0]} · {DIR_LABEL.get(k[1], k[1])}")
        e = g["std"].fillna(0)
        ax.fill_between(g.index, g["mean"] - e, g["mean"] + e, color=c,
                        alpha=0.13, linewidth=0)
        ax.annotate(f"{g['mean'].iloc[-1]:.1f}", (g.index[-1], g["mean"].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    ax.set_xlabel("Share of dictionary word pairs mixed into training data (%)")
    ax.set_ylabel("Test-set BLEU")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.margins(x=0.08)
    ax.legend(loc="lower right")
    ax.set_title("Scaling curve: BLEU vs. amount of explicit lexical supervision",
                 loc="left", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_lexicon_scaling_curve")


# -------------------------------------------------------- data contraction ---
def plot_data_contraction(out_dir: Path, df: pd.DataFrame):
    plt = _mpl()
    train_pct = {"80:10:10": 80, "70:10:20": 70, "60:10:30": 60}
    d = df[df["split"].isin(train_pct)].copy()
    if d["split"].nunique() < 2:
        return
    d["trpct"] = d["split"].map(train_pct)
    d["tag"] = _tag(d)

    dirs = sorted(d["direction"].unique())
    series: Dict[Any, Dict[str, pd.Series]] = {}
    for t in sorted(d["tag"].unique()):
        for st in sorted(d["strategy"].unique()):
            per_dir = {}
            for dr in dirs:
                g = d[(d["tag"] == t) & (d["strategy"] == st)
                      & (d["direction"] == dr)].groupby("trpct")["bleu"].mean()
                if len(g) >= 2:
                    per_dir[dr] = g.sort_index()
            if per_dir:
                series[(t, st)] = per_dir
    if not series:
        return
    cmap, mmap, keys = assign_colors(list(series))

    fig, axes = plt.subplots(1, len(dirs), figsize=(4.8 * len(dirs), 3.7),
                             squeeze=False, sharey=True)
    for i, dr in enumerate(dirs):
        ax = _clean(axes[0][i])
        for k in keys:
            g = series[k].get(dr)
            if g is None:
                continue
            ax.plot(g.index, g.values, color=cmap[k], lw=2.0, marker=mmap[k],
                    markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.0,
                    label=f"{k[0]} · {STRATEGY_LABEL.get(k[1], k[1])}")
        ax.set_xticks([60, 70, 80])
        ax.invert_xaxis()
        ax.set_xlabel("Training-data proportion (%)")
        if i == 0:
            ax.set_ylabel("Test-set BLEU")
        ax.set_title(DIR_LABEL.get(dr, dr), loc="left", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Robustness under training-corpus contraction", x=0.01,
                 ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    _save(fig, out_dir, "fig_data_contraction")


# --------------------------------------------------------- cognate & tone ---
def plot_cognate_and_tone(out_dir: Path, df: pd.DataFrame, main_split: str):
    plt = _mpl()
    cols = ["bleu_cognate_low", "bleu_cognate_mid", "bleu_cognate_high"]
    d = df[(df["split"] == main_split) & df["strategy"].isin(HEADLINE)].copy()
    if len(d) == 0 or not all(c in d.columns for c in cols):
        return
    d["tag"] = _tag(d) + " · " + d["strategy"].map(lambda s: STRATEGY_SHORT.get(s, s))
    dirs = sorted(d["direction"].unique())

    # --- cognate bands: the SAME column order in every panel for comparison ---
    g = d.groupby(["direction", "tag"])[cols + ["tone_accuracy"]].mean().reset_index()
    order = (g.groupby("tag")["bleu_cognate_high"].mean()
              .sort_values(ascending=False).index.tolist())
    names = ["Low cognate density", "Mid cognate density", "High cognate density"]

    fig, axes = plt.subplots(len(dirs), 1,
                             figsize=(min(13.5, max(7, 0.62 * len(order))),
                                      3.5 * len(dirs)),
                             squeeze=False, sharex=True)
    for i, dr in enumerate(dirs):
        ax = _clean(axes[i][0])
        sub = g[g["direction"] == dr].set_index("tag").reindex(order)
        w = 0.26
        for j, (c, nm) in enumerate(zip(cols, names)):
            xs = [k + (j - 1) * w for k in range(len(order))]
            bars = ax.bar(xs, sub[c].values, width=w * 0.9, color=ORDINAL3[j],
                          edgecolor=SURFACE, linewidth=1.4, label=nm)
            _bar_labels(ax, bars, fontsize=6.2)
        ax.set_ylabel("BLEU")
        ax.set_title(DIR_LABEL.get(dr, dr), loc="left", color=INK)
        if i == len(dirs) - 1:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7.5)
        if i == 0:
            ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.28))
    fig.suptitle("Translation quality broken down by source-sentence cognate "
                 "density", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, out_dir, "fig_cognate_density_breakdown")

    # --- tone: HORIZONTAL bars so long labels do not overlap ---
    order_t = (g.groupby("tag")["tone_accuracy"].mean()
                .sort_values().index.tolist())
    cmap, _, dirs_c = assign_colors(dirs)
    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.34 * len(order_t))))
    _clean(ax, ygrid=False, xgrid=True)
    h = 0.8 / len(dirs_c)
    for i, dr in enumerate(dirs_c):
        sub = g[g["direction"] == dr].set_index("tag").reindex(order_t)
        ys = [k + (i - (len(dirs_c) - 1) / 2) * h for k in range(len(order_t))]
        bars = ax.barh(ys, sub["tone_accuracy"].values, height=h * 0.88,
                       color=cmap[dr], edgecolor=SURFACE, linewidth=1.4,
                       label=DIR_LABEL.get(dr, dr))
        _bar_labels(ax, bars, fontsize=6.8, horizontal=True)
    ax.set_yticks(range(len(order_t)))
    ax.set_yticklabels(order_t, fontsize=7.5)
    ax.set_xlabel("Per-syllable tone accuracy (%)")
    ax.margins(x=0.10)
    ax.legend(ncol=2, loc="lower right")
    ax.set_title("Tone-mark accuracy", loc="left", color=INK)
    fig.tight_layout()
    _save(fig, out_dir, "fig_tone_accuracy")


# ------------------------------------------------------- latency / quality ---
def plot_latency_quality(out_dir: Path, df: pd.DataFrame, main_split: str):
    plt = _mpl()
    if "latency_ms_avg" not in df.columns:
        return
    d = df[(df["split"] == main_split) & df["strategy"].isin(HEADLINE)
           & df["latency_ms_avg"].notna()].copy()
    if len(d) < 3:
        return
    d["tag"] = _tag(d) + "/" + d["strategy"]
    dirs = sorted(d["direction"].unique())
    fig, axes = plt.subplots(1, len(dirs), figsize=(5.4 * len(dirs), 4.0),
                             squeeze=False, sharey=True)
    y_lo_all = float(d["bleu"].min())
    y_hi_all = float(d["bleu"].max())
    for i, dr in enumerate(dirs):
        ax = _clean(axes[0][i])
        sub = (d[d["direction"] == dr].groupby("tag")[["latency_ms_avg", "bleu"]]
               .mean().reset_index().sort_values("latency_ms_avg"))
        # Pareto front: faster and still better. Only THESE points are labelled
        # — labelling every point produces overlapping text.
        best, front = -np.inf, []
        for _, r in sub.iterrows():
            if r["bleu"] > best:
                best = r["bleu"]
                front.append(r)
        fr = pd.DataFrame(front)
        ax.scatter(sub["latency_ms_avg"], sub["bleu"], s=46, color=SERIES[0],
                   alpha=0.35, edgecolor=SURFACE, linewidth=1.2, zorder=2)
        ax.step(fr["latency_ms_avg"], fr["bleu"], where="post", color=SERIES[0],
                lw=1.6, alpha=0.55, zorder=2)
        ax.scatter(fr["latency_ms_avg"], fr["bleu"], s=68, color=SERIES[0],
                   edgecolor=SURFACE, linewidth=1.4, zorder=3)
        # Pareto points are often close on the x axis, so labels are PUSHED apart
        # until they no longer overlap and joined back with thin leader lines.
        xv = fr["latency_ms_avg"].to_numpy(dtype=float)
        yv = fr["bleu"].to_numpy(dtype=float)
        span = max(float(d["bleu"].max() - d["bleu"].min()), 1.0)
        gap = span * 0.062
        placed = yv.astype(float).copy()
        prev = -np.inf
        for idx in np.argsort(yv):
            placed[idx] = max(yv[idx], prev + gap)
            prev = placed[idx]
        y_hi_all = max(y_hi_all, float(placed.max()) + gap * 0.8)
        for k, tag in enumerate(fr["tag"].tolist()):
            ax.annotate(tag, xy=(xv[k], yv[k]), xytext=(xv[k] * 1.09, placed[k]),
                        textcoords="data", ha="left", va="center",
                        fontsize=7.2, color=INK,
                        arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8,
                                        shrinkA=0, shrinkB=3),
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE,
                                  ec="none", alpha=0.85))
        ax.set_xscale("log")
        ax.margins(x=0.18)
        ax.set_xlabel("Mean latency per sentence (ms, log scale)")
        if i == 0:
            ax.set_ylabel("Test-set BLEU")
        ax.set_title(DIR_LABEL.get(dr, dr), loc="left", color=INK)
    pad = max(y_hi_all - y_lo_all, 1.0) * 0.05
    axes[0][0].set_ylim(y_lo_all - pad, y_hi_all + pad)
    fig.suptitle("Quality vs. inference speed (batch = 1, beam = 4) — "
                 "only Pareto-optimal systems are labelled",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "fig_quality_latency_tradeoff")


# ------------------------------------------------------------- forest plot ---
def plot_significance(out_dir: Path):
    f = out_dir / "significance.csv"
    if not f.exists():
        return
    plt = _mpl()
    s = pd.read_csv(f)
    s = s[s["comparison"] == "vs_same_model_base"]
    if len(s) == 0:
        return
    s = s.sort_values("delta").tail(28).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.6, max(3.0, 0.28 * len(s) + 0.8)))
    _clean(ax, ygrid=False, xgrid=True)
    for i, r in s.iterrows():
        sig = bool(r["significant_p05"])
        c, a = (SERIES[0], 1.0) if sig else (INK2, 0.45)
        ax.plot([r["ci95_low"], r["ci95_high"]], [i, i], color=c, lw=2.0,
                alpha=a, solid_capstyle="round")
        ax.plot([r["delta"]], [i], marker="o", markersize=6, color=c,
                markeredgecolor=SURFACE, markeredgewidth=1.0, alpha=a, zorder=3)
    ax.axvline(0, color=INK2, lw=1.0, ls=":")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["run_id"].str.replace("__", " · ", regex=False), fontsize=6.5)
    ax.set_ylim(-0.8, len(s) - 0.2)
    ax.set_xlabel("Δ BLEU vs. the 'base' strategy of the same model "
                  "(95% confidence interval)")
    handles = [plt.Line2D([], [], color=SERIES[0], lw=2.4, marker="o", markersize=6,
                          markeredgecolor=SURFACE, label="Significant (p < 0.05)"),
               plt.Line2D([], [], color=INK2, lw=2.4, alpha=0.45, marker="o",
                          markersize=6, markeredgecolor=SURFACE,
                          label="Not significant")]
    ax.set_title("Paired bootstrap resampling: does the gain exceed seed noise?",
                 loc="left", color=INK)
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    _save(fig, out_dir, "fig_bootstrap_significance")


# ------------------------------------------------------------ corpus stats ---
def plot_corpus_stats(out_dir: Path):
    f = out_dir / "corpus_analysis.json"
    if not f.exists():
        return
    ca = json.loads(f.read_text(encoding="utf-8"))
    h = ca.get("histograms")
    if not h:
        return
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))
    cmap, mmap, _ = assign_colors(["Muong", "Vietnamese"])

    ax = _clean(axes[0])
    for k, lbl in [("len_muong", "Muong"), ("len_viet", "Vietnamese")]:
        e, c = np.array(h[k]["edges"]), np.array(h[k]["counts"], dtype=float)
        ctr = (e[:-1] + e[1:]) / 2
        ax.plot(ctr, c / max(c.sum(), 1) * 100, color=cmap[lbl], lw=2.0,
                marker=mmap[lbl], markersize=4, markeredgecolor=SURFACE,
                markeredgewidth=0.7, label=lbl)
    ax.set_xlabel("Sentence length (syllables)")
    ax.set_ylabel("Share of sentence pairs (%)")
    ax.legend()
    ax.set_title("Sentence-length distribution", loc="left", color=INK)

    for ax, key, xlab, ttl, col in [
            (axes[1], "cognate", "Cognate density of the sentence pair",
             "Vietnamese–Muong lexical proximity", SERIES[0]),
            (axes[2], "charsim", "Character similarity, Muong ↔ Vietnamese",
             "Orthographic overlap", SERIES[2])]:
        _clean(ax)
        e, c = np.array(h[key]["edges"]), np.array(h[key]["counts"], dtype=float)
        ax.bar((e[:-1] + e[1:]) / 2, c / max(c.sum(), 1) * 100,
               width=(e[1] - e[0]) * 0.9, color=col, edgecolor=SURFACE, linewidth=1.2)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Share of sentence pairs (%)")
        ax.set_title(ttl, loc="left", color=INK)
    m = ca.get("cognate_density_mean")
    if m is not None:
        axes[1].axvline(m, color=INK2, ls=":", lw=1.4)
        _note(axes[1], m, axes[1].get_ylim()[1] * 0.88,
              f"mean = {m:.2f}", ha="center")

    fig.suptitle("Quantitative characterisation of the bilingual corpus "
                 "(replaces the word-cloud figure)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "fig_corpus_statistics")


def make_plots(out_dir: Path, main_split: str = "80:10:10"):
    """Draw every figure. Figures are independent: one failure does not block the others."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is not installed — skipping plots (pip install matplotlib).")
        return
    raw = out_dir / "summary_raw.csv"
    df = pd.read_csv(raw) if raw.exists() else pd.DataFrame()
    if "from_scratch" in df.columns:
        df["from_scratch"] = df["from_scratch"].astype(bool)
    jobs = [
        ("training curves", lambda: plot_training_curves(out_dir, df)),
        ("main BLEU comparison", lambda: plot_main_comparison(out_dir, df, main_split)),
        ("dictionary scaling curve", lambda: plot_scaling_curve(out_dir, df, main_split)),
        ("data contraction", lambda: plot_data_contraction(out_dir, df)),
        ("cognate & tone", lambda: plot_cognate_and_tone(out_dir, df, main_split)),
        ("quality vs latency", lambda: plot_latency_quality(out_dir, df, main_split)),
        ("significance", lambda: plot_significance(out_dir)),
        ("corpus statistics", lambda: plot_corpus_stats(out_dir)),
    ]
    for name, fn in jobs:
        if len(df) == 0 and name != "corpus statistics":
            continue
        try:
            fn()
        except Exception as e:
            print(f"  [figure] skipped '{name}': {type(e).__name__}: {e}")
    figs = sorted((out_dir / "figures").glob("*.pdf")) \
        if (out_dir / "figures").exists() else []
    print(f"Drew {len(figs)} figures (vector PDF + PNG) in {out_dir/'figures'}")
    for f in figs:
        print(f"    {f.name}")


# ==============================================================================
# SECTION 11 — Self-test (no GPU and no real data needed)
# ==============================================================================

def self_test(tmp: Path) -> bool:
    print("=" * 78)
    print(" SELF-TEST — checks the whole pipeline without a GPU")
    print("=" * 78)
    rng = random.Random(0)
    tmp.mkdir(parents=True, exist_ok=True)

    vi_words = ["tôi", "anh", "chị", "đi", "ăn", "cơm", "nhà", "rồi", "không",
                "này", "một", "rau", "húng", "cây", "ông", "nội", "cho", "cháu"]
    mu_map = {"tôi": "ho", "anh": "da", "chị": "nả", "đi": "ti", "ăn": "ăn",
              "cơm": "cơm", "nhà": "nhà", "rồi": "rồi", "không": "chăng",
              "này": "nì", "một": "môch", "rau": "rau", "húng": "húng",
              "cây": "cẳl", "ông": "ông", "nội": "nôl", "cho": "cho",
              "cháu": "cháo"}

    rows = []
    for i in range(600):
        n = rng.randint(3, 9)
        v = [rng.choice(vi_words) for _ in range(n)]
        m = [mu_map[w] for w in v]
        rows.append({COL_V: " ".join(v), COL_M: " ".join(m)})
    for i in range(40):                                  # deliberate near-duplicates
        rows.append(dict(rows[i]))
        rows.append({COL_V: rows[i][COL_V] + " rồi", COL_M: rows[i][COL_M] + " rồi"})
    df_s = pd.DataFrame(rows)
    df_w = pd.DataFrame([{COL_V: k, COL_M: v} for k, v in mu_map.items()])

    sf, wf = tmp / "s.csv", tmp / "w.csv"
    df_s.to_csv(sf, index=False)
    df_w.to_csv(wf, index=False)

    ok = True

    # 1) dedup + group + split
    d, stats = dedup_and_group(df_s)
    sp = grouped_split(d, SPLIT_SCHEMES["80:10:10"], 42)
    tr_keys = set(sp["train"][COL_M].map(norm_text))
    te_keys = set(sp["test"][COL_M].map(norm_text))
    leak = tr_keys & te_keys
    print(f"  [1] dedup {stats}  -> train/val/test = "
          f"{len(sp['train'])}/{len(sp['val'])}/{len(sp['test'])}, "
          f"exact leaks = {len(leak)}")
    ok &= (len(leak) == 0)

    # 2) data strategies
    for st in STRATEGIES:
        aug, info = build_train_frame(sp["train"], df_w, st, 42)
        assert len(aug) >= len(sp["train"]), st
    print(f"  [2] {len(STRATEGIES)} data strategies: OK")

    # 3) baselines + metrics
    srcs = sp["test"][COL_M].tolist()
    refs = sp["test"][COL_V].tolist()
    hyp_copy = baseline_copy(srcs)
    rb = RuleBasedTranslator(df_w, COL_M, COL_V)
    hyp_rule = [rb.translate(s) for s in srcs]
    m_copy = compute_all_metrics(srcs, hyp_copy, refs)
    m_rule = compute_all_metrics(srcs, hyp_rule, refs)
    print(f"  [3] COPY  BLEU={m_copy['bleu']:.2f} chrF++={m_copy['chrf2']:.2f} "
          f"tone_acc={m_copy['tone_accuracy']:.1f}")
    print(f"      RULE  BLEU={m_rule['bleu']:.2f} chrF++={m_rule['chrf2']:.2f} "
          f"coverage={rb.coverage(srcs):.1f}%")
    ok &= m_rule["bleu"] > m_copy["bleu"]

    # 4) bootstrap
    st = paired_bootstrap(hyp_rule, hyp_copy, refs, n_samples=200)
    print(f"  [4] paired bootstrap: delta={st['delta']:.2f} p={st['p_value']:.4f} "
          f"CI95=[{st['ci95_low']:.2f}, {st['ci95_high']:.2f}]")
    ok &= st["p_value"] < 0.05

    # 5) tones & cognates
    print(f"  [5] tone('cẳl')={syllable_tone('cẳl')}  "
          f"strip('húng')={strip_tones('húng')}  "
          f"cognate_density={cognate_density('ho ti ăn cơm', 'tôi đi ăn cơm'):.3f}")
    ok &= syllable_tone("cẳl") == "hoi"
    ok &= strip_tones("húng") == "hung"

    # 6) job CPU end-to-end
    for d_ in ("results", "predictions", "work"):
        (tmp / d_).mkdir(parents=True, exist_ok=True)
    r = run_cpu_job({"job": dict(run_id="selftest_copy", model="copy",
                                 direction="m2v", strategy="none",
                                 split="80:10:10", seed=42, from_scratch=False),
                     "out_dir": str(tmp), "sentence_file": str(sf),
                     "word_file": str(wf), "jaccard_thr": 0.8})
    print(f"  [6] run_cpu_job -> {r}")
    ok &= r["ok"]

    # 7) planning
    class A:
        experiments = EXPERIMENTS
        seeds = [42, 43]
        ablation_seeds = 1
        variance_seeds = 5
        variance_model = "vit5-base"
        ablation_model = "vit5-base"
        models = ALL_MODELS
        scratch_models = ALL_MODELS
        include_transformer_scratch = True
        splits = list(SPLIT_SCHEMES)
        main_split = "80:10:10"
        directions = DIRECTIONS
    cj, gj = plan_jobs(A())
    n_train_est = 10290
    cost_h = sum(estimate_cost(j, n_train_est) for j in gj) * 300 / 3600
    print(f"  [7] plan: {len(cj)} CPU jobs, {len(gj)} GPU jobs "
          f"(~{cost_h:.0f} GPU hours)")
    ok &= len(cj) > 0 and len(gj) > 0

    # UNIFORMITY CHECK: within each experiment every (model, scratch) must have
    # exactly the same seeds, strategies and both directions, i.e. no table
    # has holes.
    import collections as _c
    for exp in sorted({j["experiment"] for j in gj}):
        cells = _c.defaultdict(lambda: _c.defaultdict(set))
        for j in gj:
            if j["experiment"] != exp:
                continue
            key = (j["model"], j["from_scratch"])
            cells[key]["seed"].add(j["seed"])
            cells[key]["strategy"].add(j["strategy"])
            cells[key]["direction"].add(j["direction"])
            cells[key]["split"].add(j["split"])
        sig = {k: tuple(sorted(map(str, v["seed"]))) + tuple(sorted(v["strategy"]))
                  + tuple(sorted(v["direction"])) + tuple(sorted(v["split"]))
               for k, v in cells.items()}
        uniform = len(set(sig.values())) == 1
        print(f"      {exp:<18} {len(cells)} rows, uniform grid: "
              f"{'OK' if uniform else 'NOT UNIFORM'}")
        ok &= uniform

    # 8) corpus analysis + aggregation
    corpus_analysis(d, df_w, tmp, stats)
    aggregate(tmp)
    print("  [8] plots:")
    make_plots(tmp, "80:10:10")

    print("=" * 78)
    print(" SELF-TEST:", "PASS ✅" if ok else "FAIL ❌")
    print("=" * 78)
    return ok


# ==============================================================================
# SECTION 12 — main
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description="VietMuong experiment runner (Vietnamese <-> Muong MT)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sentence_file", default=None,
                   help="Local sentence-pair CSV (columns SentenceM, SentenceV). "
                        "Default: download from the Hugging Face Hub.")
    p.add_argument("--word_file", default=None,
                   help="Local lexical-pair CSV (columns WordM, WordV). "
                        "Default: download from the Hugging Face Hub.")
    p.add_argument("--hf_revision", default=HF_DATASET_REVISION,
                   help=f"Revision of {HF_DATASET_ID} on the Hub (default: the one used for the paper)")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--experiments", nargs="+", default=EXPERIMENTS,
                   choices=EXPERIMENTS,
                   help="Experiments to run. Each experiment is a FULL grid and is "
                        "run either completely or not at all.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Seeds for E1, applied to EVERY model in E1. The paper "
                        "uses 42 43 44 (mean ± std in Table II). With >= 3 seeds "
                        "E5_seed_variance is skipped automatically.")
    p.add_argument("--ablation_seeds", type=int, default=3,
                   help="Number of seeds for E3/E4 (same within each experiment). "
                        "The paper uses 3 (seeds 42-44) on --ablation_model.")
    p.add_argument("--variance_seeds", type=int, default=5,
                   help="Number of seeds for E5 (seed-variance study)")
    p.add_argument("--variance_model", default="vit5-base", choices=list(MODEL_REGISTRY),
                   help="Model used for E5")
    p.add_argument("--ablation_model", default="vit5-base", choices=list(MODEL_REGISTRY),
                   help="Model used for E3 and E4 (ViT5-base in the paper)")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=list(MODEL_REGISTRY),
                   help="Models of E1 (main table)")
    p.add_argument("--scratch_models", nargs="+", default=ALL_MODELS,
                   choices=list(MODEL_REGISTRY),
                   help="[NO EFFECT] E2_scratch is not part of EXPERIMENTS (no "
                        "model is trained from scratch). Kept only so that old "
                        "commands still parse; add 'E2_scratch' back to "
                        "EXPERIMENTS in the code to re-enable it.")
    p.add_argument("--include_transformer_scratch", action="store_true", default=True,
                   help="[NO EFFECT] see --scratch_models above.")
    p.add_argument("--no_transformer_scratch", dest="include_transformer_scratch",
                   action="store_false")
    p.add_argument("--splits", nargs="+", default=list(SPLIT_SCHEMES),
                   choices=list(SPLIT_SCHEMES))
    p.add_argument("--main_split", default="80:10:10", choices=list(SPLIT_SCHEMES))
    p.add_argument("--directions", nargs="+", default=DIRECTIONS, choices=DIRECTIONS)
    p.add_argument("--only_models", nargs="+", default=None,
                   help="CALIBRATION/DEBUGGING ONLY. Filtering by model BREAKS "
                        "the uniform grid — results cannot be used to build the "
                        "paper's tables. To restrict models properly use "
                        "--models / --scratch_models.")
    p.add_argument("--only_run_ids", nargs="+", default=None,
                   help="Run ONLY these run_ids (filters gpu_jobs after planning "
                        "and drops all cpu_jobs). Each run keeps exactly the same "
                        "protocol as in the full grid.")
    p.add_argument("--epochs_override", type=int, default=None,
                   help="Force the number of epochs for EVERY model (quick tests only)")
    p.add_argument("--max_len_override", type=int, default=None,
                   help="Force the max sequence length for EVERY model")
    p.add_argument("--latency_n", type=int, default=100,
                   help="Number of test sentences for latency (batch=1, beam=4). "
                        "Measured only for the first seed (Table II uses 100).")
    p.add_argument("--val_subset", type=int, default=500,
                   help="Validation sentences used to SELECT the checkpoint each "
                        "epoch (paper: 500 of 1,284). 0 = full validation set. "
                        "The final TEST evaluation is not affected.")
    p.add_argument("--sec_per_cost_prior", type=float, default=260.0,
                   help="Initial seconds-per-cost-unit estimate, only for the ETA "
                        "before real timings exist; it recalibrates after the "
                        "first finished run.")
    p.add_argument("--cpu_workers", type=int, default=max(1, available_cpus() // 4),
                   help="Parallel processes for CPU jobs (default = 1/4 of the available cores)")
    p.add_argument("--jaccard_thr", type=float, default=0.8,
                   help="Jaccard threshold for near-duplicate grouping (paper: 0.8)")
    p.add_argument("--exclude_test_words", action="store_true",
                   help="STRICT ablation: drop dictionary pairs that appear in the test set (not used in the paper)")
    p.add_argument("--gpus", default=None,
                   help="GPUs to use IN PARALLEL, e.g. '0,1'. One process per GPU "
                        "pulling jobs from a shared queue. If empty, the script "
                        "uses torch.cuda.device_count(); if CUDA_VISIBLE_DEVICES "
                        "is set, these indices are RELATIVE to that set (0,1,...).")
    p.add_argument("--gpu", default=None,
                   help="[legacy] A single GPU. Same as --gpus <id>.")
    p.add_argument("--workers_per_gpu", type=int, default=1,
                   help="Jobs to run IN PARALLEL on EACH GPU (default 1, as in the "
                        "paper). VRAM is split between workers, so 2 only suits "
                        "small models — the script warns about models that do not "
                        "fit. Note: latency is measured inside each run.")
    p.add_argument("--keep_checkpoints", action="store_true",
                   help="Keep checkpoints after a run finishes (default: DELETE; "
                        "100+ runs x several GB would fill the disk)")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore unfinished checkpoints and train from scratch")
    p.add_argument("--torch_threads", type=int, default=None,
                   help="CPU threads for torch/BLAS. Default = min(available cores, 16), "
                        "detected correctly under `taskset -c 0-15`.")
    p.add_argument("--budget_days", type=float, default=None,
                   help="Budget in WALL-CLOCK days. The script drops whole "
                        "low-priority experiments to fit. Multiplied by the number of GPUs.")
    p.add_argument("--budget_gpu_hours", type=float, default=None,
                   help="Budget in GPU hours. Overrides --budget_days.")
    p.add_argument("--dry_run", action="store_true", help="Only print the plan")
    p.add_argument("--report_only", action="store_true", help="Only aggregate existing results")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip plotting")
    p.add_argument("--plot_only", action="store_true",
                   help="Only redraw plots from an existing summary_raw.csv")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    for d in ("results", "predictions", "work", "logs", "curves", "figures"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    if args.self_test:
        sys.exit(0 if self_test(Path("./_vimu_selftest")) else 1)

    if args.report_only or args.plot_only:
        if not args.plot_only:
            aggregate(out_dir)
        if not args.no_plots:
            make_plots(out_dir, args.main_split)
        return

    if args.gpus is None and args.gpu is not None:
        args.gpus = str(args.gpu)
    env = setup_compute_env(None, args.torch_threads)
    if args.gpus is None:
        args.gpus = detect_gpus()
    print(f"GPU  : using [{args.gpus}] in parallel"
          + (f"  (CUDA_VISIBLE_DEVICES={env['cuda_visible_devices']} "
             f"set by the environment; the indices above are relative to it)"
             if env["cuda_from_env"] else ""))
    print(f"CPU  : {env['cpu_affinity']} usable cores (taskset affinity), "
          f"{env['torch_threads']} threads for torch/BLAS")

    # ---------------- Data ----------------
    print("Loading data...")
    args.sentence_file, args.word_file = resolve_data_files(
        args.sentence_file, args.word_file, args.hf_revision)
    df_s_raw, df_w = load_frames(args.sentence_file, args.word_file)
    df_s, dstats = dedup_and_group(df_s_raw, jaccard_thr=args.jaccard_thr)
    print(f"  Sentences: {dstats}")
    print(f"  Dictionary pairs: {len(df_w)}")
    ca = corpus_analysis(df_s, df_w, out_dir, dstats)
    print(f"  Mean cognate density: {ca['cognate_density_mean']:.3f} | "
          f"length ratio M/V: {ca['length_ratio_M_over_V']:.3f} | "
          f"vocab overlap: {ca['vocab_overlap_pct']:.1f}%")

    # ---------------- Plan ----------------
    cpu_jobs, gpu_jobs = plan_jobs(args)
    if args.only_models:
        print("\n  !! --only_models is on: the experiment grid is NO LONGER "
              "uniform.\n     Use it only to calibrate speed; do not build the paper's "
              "tables from it.\n     To restrict models properly: --models / "
              "--scratch_models.")
        cpu_jobs = [j for j in cpu_jobs if j["model"] in args.only_models]
        gpu_jobs = [j for j in gpu_jobs if j["model"] in args.only_models]

    if args.only_run_ids:
        wanted_ids = list(dict.fromkeys(args.only_run_ids))
        planned = {j["run_id"] for j in gpu_jobs}
        missing = [r for r in wanted_ids if r not in planned]
        if missing:
            raise SystemExit(f"--only_run_ids: not in the plan: {missing}")
        cpu_jobs = []
        gpu_jobs = [j for j in gpu_jobs if j["run_id"] in set(wanted_ids)]
        print(f"  [--only_run_ids] running only {len(gpu_jobs)} runs: "
              + ", ".join(j["run_id"] for j in gpu_jobs))

    if not args.overwrite:
        done = {f.stem for f in (out_dir / "results").glob("*.json")}
        cpu_jobs = [j for j in cpu_jobs if j["run_id"] not in done]
        gpu_jobs = [j for j in gpu_jobs if j["run_id"] not in done]

    if args.epochs_override:
        for s in MODEL_REGISTRY.values():
            s.epochs = args.epochs_override

    n_train_est = int(len(df_s) * SPLIT_SCHEMES[args.main_split][0])
    costs = {j["run_id"]: estimate_cost(j, n_train_est) for j in gpu_jobs}
    phys_gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    n_phys = max(len(phys_gpus), 1)
    wpg = max(1, int(args.workers_per_gpu))
    # Worker labels: with 1 worker/GPU keep the GPU name (dashboard as before);
    # with several workers/GPU append '#k' to tell them apart on the dashboard.
    if wpg == 1:
        gpus = list(phys_gpus)
    else:
        gpus = [f"{g}#{k+1}" for g in phys_gpus for k in range(wpg)]
    n_gpu = max(len(gpus), 1)          # number of WORKERS (used to split CPU threads)

    # ---------------- Budget: drop low-priority experiments ----------------
    dropped: List[Dict] = []
    budget_h = args.budget_gpu_hours
    if budget_h is None and args.budget_days is not None:
        budget_h = args.budget_days * 24 * n_phys
    dropped_exps: List[str] = []
    if budget_h is not None:
        gpu_jobs, dropped, dropped_exps = apply_budget(
            gpu_jobs, costs, budget_h, args.sec_per_cost_prior, EXPERIMENTS)

    # LPT: expensive jobs first => GPUs finish at about the same time
    gpu_jobs.sort(key=lambda j: -costs[j["run_id"]])

    def hours(js):
        return sum(costs[j["run_id"]] for j in js) * args.sec_per_cost_prior / 3600

    print(f"\nPLAN: {len(cpu_jobs)} CPU jobs (parallel, {args.cpu_workers} "
          f"workers), {len(gpu_jobs)} GPU jobs on {n_phys} GPU(s)"
          + (f" x {wpg} worker/GPU" if wpg > 1 else "")
          + f" [{', '.join(gpus)}]")

    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for j in gpu_jobs:
        by_exp[j["experiment"]].append(j)
    for exp in EXPERIMENTS:
        js = by_exp.get(exp)
        if not js:
            continue
        rows = sorted({f"{j['model']}{'-scratch' if j['from_scratch'] else ''}"
                       for j in js})
        sds = sorted({j["seed"] for j in js})
        sts = sorted({j["strategy"] for j in js},
                     key=lambda x: STRATEGIES.index(x) if x in STRATEGIES else 99)
        sps = sorted({j["split"] for j in js})
        reused = ""
        if exp == "E3_strategy" and "base" not in sts:
            reused = "  (the 'base'/'word' cells reuse E1 results and are not run again)"
        print(f"\n  [{exp}] {EXPERIMENT_TITLE[exp]}{reused}")
        print(f"      grid : {len(rows)} models x {len(sts)} strategies x "
              f"{len(sorted({j['direction'] for j in js}))} directions x "
              f"{len(sds)} seed × {len(sps)} split = {len(js)} run"
              f"   ({hours(js):.0f} GPU hours)")
        print(f"      models    : {', '.join(rows)}")
        print(f"      strategies: {', '.join(sts)}")
        print(f"      seed      : {', '.join(map(str, sds))}"
              f"   |   split: {', '.join(sps)}")

    tot_h = hours(gpu_jobs)
    print(f"\n    TOTAL {len(gpu_jobs)} runs, {tot_h:.0f} GPU hours"
          f"  =>  on {n_phys} GPU(s): ~{tot_h/n_phys:.0f} hours ~ "
          f"{tot_h/n_phys/24:.1f} days")
    if wpg > 1:
        print(f"    ({wpg} workers per GPU do NOT halve the time: they share "
              f"one\n     GPU and only fill its idle time; in practice ~10-30% "
              f"faster.)")

    # ---- VRAM warning when several workers share one GPU ----
    if wpg > 1:
        total_vram = 24.0                      # A5000 default; detected when possible
        try:
            _t = _torch()
            if _t.cuda.is_available():
                total_vram = _t.cuda.get_device_properties(0).total_memory / 2 ** 30
        except Exception:
            pass
        cap = total_vram * 0.96 / wpg          # the exact cap set for each worker
        used = sorted({j["model"] for j in gpu_jobs})
        safe, tight, risky = [], [], []
        for k in used:
            v = estimate_vram_gb(MODEL_REGISTRY[k])
            if v is None:
                continue
            if v > cap:
                risky.append((k, v))
            elif v > 0.8 * cap:                # less than 20% safety margin
                tight.append((k, v))
            else:
                safe.append((k, v))
        print(f"\n    VRAM: {total_vram:.1f} GB / {wpg} worker => ~{cap:.1f} GB "
              f"per worker (estimate, excluding beam search):")
        if safe:
            print("      fits easily   : "
                  + ", ".join(f"{k} ~{v:.1f}GB" for k, v in safe))
        if tight:
            print("      TIGHT         : "
                  + ", ".join(f"{k} ~{v:.1f}GB" for k, v in tight)
                  + "\n         (<20% margin; beam=4 decoding can overflow "
                    "and trigger a configuration fallback)")
        if risky:
            print("      DOES NOT FIT  : "
                  + ", ".join(f"{k} ~{v:.1f}GB" for k, v in risky))
        if tight or risky:
            print("      => run those models alone with --workers_per_gpu 1.")
    print(f"    (ESTIMATE. Early stopping usually cuts another 25-35%, and this "
          f"number\n     recalibrates after the first finished run. To be "
          f"sure\n     before committing: run one calibration job "
          f"--only_models vit5-base --seeds 42)")
    if dropped_exps:
        print(f"\n    NOT RUN because of the budget ({hours(dropped):.0f} GPU hours) — "
              f"whole experiments are dropped, never single cells, so the "
              f"remaining tables stay complete:")
        for exp in dropped_exps:
            js = [j for j in dropped if j["experiment"] == exp]
            print(f"      - {exp:<18} {EXPERIMENT_TITLE[exp]}  "
                  f"({len(js)} runs, {hours(js):.0f} hours)")
        print(f"      Run them later: drop --budget_days, or "
              f"--experiments {' '.join(dropped_exps)}")
        (out_dir / "plan_dropped.json").write_text(
            json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        (out_dir / "plan.json").write_text(
            json.dumps({"cpu": cpu_jobs, "gpu": gpu_jobs}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n--dry_run: plan written to {out_dir/'plan.json'}. Exiting.")
        return

    first_seed = args.seeds[0] if args.seeds else 42
    for j in gpu_jobs:
        j["latency_n"] = args.latency_n
        j["val_subset"] = args.val_subset
        j["measure_latency"] = (j["seed"] == first_seed)
        j["keep_checkpoints"] = args.keep_checkpoints
        j["fresh"] = args.fresh

    dash = Dashboard(gpu_jobs, cpu_jobs, costs, gpus, args.sec_per_cost_prior)
    dash.start()

    # ---------------- CPU: song song ----------------
    futures = {}
    pool = None
    if cpu_jobs:
        pool = ProcessPoolExecutor(max_workers=args.cpu_workers)
        for j in cpu_jobs:
            futures[pool.submit(run_cpu_job, {
                "job": j, "out_dir": str(out_dir),
                "sentence_file": args.sentence_file, "word_file": args.word_file,
                "jaccard_thr": args.jaccard_thr})] = j["run_id"]

    # ---------------- GPU: one process per GPU, pulling from a shared queue ----
    procs = []
    if gpu_jobs:
        ctx = mp.get_context("spawn")
        job_q, status_q = ctx.Queue(), ctx.Queue()
        for j in gpu_jobs:
            job_q.put(j)
        cfg = {"out_dir": str(out_dir), "sentence_file": args.sentence_file,
               "word_file": args.word_file, "jaccard_thr": args.jaccard_thr,
               "exclude_test_words": args.exclude_test_words,
               "torch_threads": args.torch_threads, "n_gpu": n_gpu,
               "workers_per_gpu": wpg,
               "epochs_override": args.epochs_override,
               "max_len_override": args.max_len_override}
        for g in gpus:
            pr = ctx.Process(target=gpu_worker, args=(g, job_q, status_q, cfg),
                             daemon=False)
            pr.start()
            procs.append(pr)

        alive = len(procs)
        while alive > 0:
            try:
                ev = status_q.get(timeout=1.0)
            except Exception:
                for f in [f for f in list(futures) if f.done()]:
                    r = f.result()
                    dash.cpu_done += 1
                    if not r.get("ok"):
                        dash.errors.append(f"[CPU] {r.get('run_id')}: "
                                           f"{str(r.get('error'))[:160]}")
                    futures.pop(f, None)
                dash.refresh()
                continue
            g = ev.get("gpu")
            if ev["type"] == "start":
                dash.cur[g] = ev["run_id"]
                dash.msg[g] = "starting"
            elif ev["type"] == "status":
                dash.msg[g] = ev["msg"]
            elif ev["type"] == "done":
                dash.note_finished(ev["run_id"], ev.get("bleu"), ev["secs"])
                if ev.get("error"):
                    dash.errors.append(f"[GPU{g}] {ev['run_id']}: {ev['error'][:160]}")
                dash.cur[g] = None
                dash.msg[g] = "waiting"
            elif ev["type"] in ("exit", "fatal"):
                if ev["type"] == "fatal":
                    dash.errors.append(f"[GPU{g}] FATAL: {ev.get('error')}")
                dash.msg[g] = "xong"
                dash.cur[g] = None
                alive -= 1
            dash.refresh()
        for pr in procs:
            pr.join(timeout=60)

    # wait for the CPU jobs
    for f in as_completed(list(futures)):
        r = f.result()
        dash.cpu_done += 1
        if not r.get("ok"):
            dash.errors.append(f"[CPU] {r.get('run_id')}: {str(r.get('error'))[:160]}")
        dash.refresh()
    if pool:
        pool.shutdown(wait=True)
    dash.stop()

    print("\nDone. Aggregating...")
    aggregate(out_dir)
    if not args.no_plots:
        print("Plotting...")
        make_plots(out_dir, args.main_split)


if __name__ == "__main__":
    main()