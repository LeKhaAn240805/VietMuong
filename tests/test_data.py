# -*- coding: utf-8 -*-
"""Data-pipeline checks against the numbers reported in the paper (Sec. III-D).

Run:  python -m pytest -q tests/test_data.py
(downloads the dataset from the Hugging Face Hub on first use; no GPU needed)
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location("vimu", ROOT / "run_vimu_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vimu"] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load_runner()


@pytest.fixture(scope="module")
def frames():
    sentence_file, word_file = R.resolve_data_files(None, None)
    return R.load_frames(sentence_file, word_file)


def test_counts(frames):
    df_s, df_w = frames
    assert len(df_s) == 12839          # deduplicated sentence pairs
    assert len(df_w) == 15377          # lexical pairs


def test_groups_and_splits(frames):
    df_s, _ = frames
    d, stats = R.dedup_and_group(df_s, jaccard_thr=0.8)
    assert stats["after_exact_dedup"] == 12839
    assert stats["n_groups"] == 12800
    for seed in (42, 43, 44, 45, 46):
        sp = R.grouped_split(d, R.SPLIT_SCHEMES["80:10:10"], seed)
        assert [len(sp[k]) for k in ("train", "val", "test")] == [10272, 1284, 1283]
        # grouped_split drops the group column; recover each pair's group from d
        group_of = {(m, v): g for m, v, g in zip(d[R.COL_M], d[R.COL_V], d["group"])}
        g = {k: {group_of[(m, v)] for m, v in zip(sp[k][R.COL_M], sp[k][R.COL_V])}
             for k in ("train", "val", "test")}
        assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"])


def test_strategy_sizes(frames):
    df_s, df_w = frames
    d, _ = R.dedup_and_group(df_s, jaccard_thr=0.8)
    train = R.grouped_split(d, R.SPLIT_SCHEMES["80:10:10"], 42)["train"]
    sizes = {}
    for strategy in ("base", "word_p25", "word_p50", "word_p75", "word", "ctrl_shuffle", "ctrl_dup"):
        frame, _info = R.build_train_frame(train, df_w, strategy, seed=42)
        sizes[strategy] = len(frame)
    # Table III, "Instances" column
    assert sizes == {"base": 10272, "word_p25": 14116, "word_p50": 17960, "word_p75": 21805,
                     "word": 25649, "ctrl_shuffle": 25649, "ctrl_dup": 25649}
