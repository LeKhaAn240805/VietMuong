# -*- coding: utf-8 -*-
"""Checks for the ViT5 vocabulary expansion used by the token / token_avg runs.

Run:  python -m pytest -q -s tests/test_vocab_expansion.py
(-s prints the numbers: checked strings, fertility, ...). Needs the ViT5-base
tokenizer and weights from the Hugging Face Hub; no GPU needed.

The script is loaded by file path; do not add an __init__.py to this folder.
"""
import importlib.util
import json
import re
import sys
import unicodedata
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
WS = re.compile(r"\s+")


def ws(s):
    return WS.sub(" ", s).strip()


@pytest.fixture(scope="module")
def data():
    df_s, df_w = R.load_frames(*R.resolve_data_files(None, None))
    d, _ = R.dedup_and_group(df_s, 0.8)
    return d, df_w


@pytest.fixture(scope="module")
def toks(data, tmp_path_factory):
    d, df_w = data
    tok0 = R.load_t5_tokenizer_v4compat("VietAI/vit5-base")
    texts = df_w[R.COL_M].tolist() + df_w[R.COL_V].tolist()
    out = tmp_path_factory.mktemp("tokvx") / "tokenizer.json"
    tokx, pieces, info = R.build_word_boundary_tokenizer(tok0, texts, out)
    return tok0, tokx, pieces, info


def _all_texts(d, df_w):
    sents_m = d[R.COL_M].tolist()
    sents_v = d[R.COL_V].tolist()
    dict_m = df_w[R.COL_M].tolist()
    dict_v = df_w[R.COL_V].tolist()
    pre = list(R.PREFIX.values())
    with_prefix = [R.PREFIX["m2v"] + s for s in sents_m] + [R.PREFIX["v2m"] + s for s in sents_v]
    return {"sent_M": sents_m, "sent_V": sents_v, "dict_M": dict_m, "dict_V": dict_v,
            "prefix_only": pre, "prefix+src": with_prefix}


def test_0_counts(toks):
    tok0, tokx, pieces, info = toks
    print(f"\n[0] len(tok0)={len(tok0)} len(tokx)={len(tokx)} new={len(pieces)} info={info}")
    assert len(tok0) == 36096
    assert info["n_new_tokens"] == len(pieces) == 2727      # Sec. IV-B: 2,727 added items
    assert len(tokx) == len(tok0) + len(pieces)


def test_1_roundtrip(data, toks):
    """decode(encode(x)) == x (whitespace-normalised, modulo the v4 clean-up that
    applies to every system). The original tokenizer already fails on a few
    sentences (characters missing from its vocabulary -> <unk>, dropped when
    decoding). Requirement: the expanded tokenizer introduces NO new round-trip
    failure (every remaining failure equals the original one and comes from <unk>)."""
    d, df_w = data
    tok0, tokx, _, _ = toks
    res = {}
    total = 0
    for name, xs in _all_texts(d, df_w).items():
        e0 = tok0(xs)["input_ids"]
        ex = tokx(xs)["input_ids"]
        d0 = tok0.batch_decode(e0, skip_special_tokens=True)
        dx = tokx.batch_decode(ex, skip_special_tokens=True)
        tgt = [ws(tok0.clean_up_tokenization(ws(unicodedata.normalize("NFKC", x)))) for x in xs]
        bad0 = [i for i in range(len(xs)) if ws(d0[i]) != tgt[i]]
        badx = [i for i in range(len(xs)) if ws(dx[i]) != tgt[i]]
        # a "new" failure = wrong decoding by the expanded tokenizer WITHOUT <unk>
        # in its own encoding (i.e. caused by the new tokens)
        new_fail = [i for i in badx if tok0.unk_token_id not in ex[i] and ws(dx[i]) != ws(d0[i])]
        unk_explained = all(tok0.unk_token_id in ex[i] or ws(dx[i]) == ws(d0[i]) for i in badx)
        fixed = [i for i in bad0 if i not in set(badx)]
        res[name] = dict(n=len(xs), fail_original=len(bad0), fail_expanded=len(badx),
                         new_failures=len(new_fail), fixed_by_expansion=len(fixed),
                         residual_all_unk=bool(unk_explained),
                         differs_from_original_output=sum(ws(dx[i]) != ws(d0[i]) for i in range(len(xs))))
        total += len(xs)
        assert not new_fail, (name, [(xs[i], dx[i]) for i in new_fail[:5]])
        assert unk_explained, name
        assert len(badx) <= len(bad0)
    print(f"\n[1] round-trip on {total} strings: {json.dumps(res, ensure_ascii=False)}")


def _word_inventory(d, df_w):
    words = set()
    for col in (R.COL_M, R.COL_V):
        for s in d[col].tolist() + df_w[col].tolist():
            words.update(str(s).split())
    words.update(" ".join(R.PREFIX.values()).split())
    return sorted(words)


def test_2_word_boundary(data, toks):
    """A new token is used only when the WHOLE word equals its syllable; every
    other word is encoded exactly as by the original tokenizer (this also checks
    the side effects of ignore_merges)."""
    d, df_w = data
    tok0, tokx, pieces, _ = toks
    old_n = len(tok0)
    words = _word_inventory(d, df_w)
    piece_of = {p: s for s, p in pieces}
    e0 = tok0(words, add_special_tokens=False)["input_ids"]
    ex = tokx(words, add_special_tokens=False)["input_ids"]
    misuse, changed, whole_word_hits = [], [], 0
    norm = tok0.backend_tokenizer.normalizer
    for w, a, b in zip(words, e0, ex):
        new_ids = [i for i in b if i >= old_n]
        key = "▁" + norm.normalize_str(w)
        if new_ids:
            ok = len(b) == 1 and tokx.convert_ids_to_tokens(b[0]) == key and key in piece_of
            if not ok:
                misuse.append((w, tokx.convert_ids_to_tokens(b)))
            else:
                whole_word_hits += 1
        elif a != b:
            changed.append((w, tok0.convert_ids_to_tokens(a), tokx.convert_ids_to_tokens(b)))
    # For every new syllable s and every word w that contains s as a PROPER
    # substring, encode(w) must not contain id(s).
    id_of = {s: tokx.convert_tokens_to_ids(p) for s, p in pieces}
    enc_x = dict(zip(words, ex))
    n_pairs = 0
    viol = []
    lw = [(w, w.lower()) for w in words]
    for s, _p in pieces:
        sid = id_of[s]
        for w, wl in lw:
            if s != wl and s in wl:
                n_pairs += 1
                if sid in enc_x[w]:
                    viol.append((s, w))
    print(f"\n[2] {len(words)} words, {n_pairs} (new syllable ⊂ word) pairs checked, violations={len(viol)}, "
          f"misuse={len(misuse)}, old words re-encoded={len(changed)}, whole-word uses={whole_word_hits}")
    assert not viol, viol[:10]
    assert not misuse, misuse[:10]
    assert not changed, changed[:10]


def test_3_single_token_and_fertility(data, toks):
    d, df_w = data
    tok0, tokx, pieces, _ = toks
    syl = [s for s, _ in pieces]
    enc = tokx(syl, add_special_tokens=False)["input_ids"]
    bad = [(s, tokx.convert_ids_to_tokens(e)) for s, e in zip(syl, enc)
           if len(e) != 1 or e[0] != tokx.convert_tokens_to_ids("▁" + s)]
    # also a single token in the middle of a sentence
    mid = tokx(["a " + s + " b" for s in syl], add_special_tokens=False)["input_ids"]
    bad_mid = [s for s, e in zip(syl, mid) if tokx.convert_tokens_to_ids("▁" + s) not in e]
    fert = {}
    for col, lab in ((R.COL_M, "Muong"), (R.COL_V, "Vietnamese")):
        xs = d[col].tolist()
        n0 = sum(len(e) for e in tok0(xs, add_special_tokens=False)["input_ids"]) / len(xs)
        nx = sum(len(e) for e in tokx(xs, add_special_tokens=False)["input_ids"]) / len(xs)
        nw = sum(len(str(x).split()) for x in xs) / len(xs)
        ex = tokx(xs, add_special_tokens=False)["input_ids"]
        share_new = sum(sum(i >= len(tok0) for i in e) for e in ex) / max(1, sum(len(e) for e in ex))
        fert[lab] = dict(n_sent=len(xs), tokens_per_sent_before=round(n0, 3),
                         tokens_per_sent_after=round(nx, 3), words_per_sent=round(nw, 3),
                         tokens_per_word_before=round(n0 / nw, 4), tokens_per_word_after=round(nx / nw, 4),
                         share_new_tokens_after=round(share_new, 5))
    print(f"\n[3] {len(syl)} new syllables: not a single token={len(bad)}, wrong mid-sentence={len(bad_mid)}")
    print(f"    fertility: {json.dumps(fert, ensure_ascii=False)}")
    assert not bad, bad[:10]
    assert not bad_mid, bad_mid[:10]


@pytest.mark.parametrize("mode", ["avg", "rand"])
def test_4_embeddings(data, toks, mode, tmp_path):
    import torch
    from transformers import AutoModelForSeq2SeqLM
    d, df_w = data
    tok0, _, pieces, _ = toks
    torch.manual_seed(0)
    model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    orig = model.get_input_embeddings().weight.detach().clone()
    texts = df_w[R.COL_M].tolist() + df_w[R.COL_V].tolist()
    tokx, info = R.expand_vocab_word_boundary(tok0, model, texts, mode, tmp_path)
    emb = model.get_input_embeddings().weight.detach()
    out = model.get_output_embeddings().weight.detach()
    old_n = len(tok0)
    assert emb.shape[0] == len(tokx) == old_n + len(pieces)
    assert torch.equal(emb[:old_n], orig), "rows of existing tokens changed"
    assert out.data_ptr() == emb.data_ptr(), "lm_head must stay tied to the embeddings"
    e0 = tokx.convert_tokens_to_ids("<extra_id_0>")
    assert e0 == tok0.convert_tokens_to_ids("<extra_id_0>") == 36095
    assert torch.equal(emb[e0], orig[e0])
    new = emb[old_n:]
    if mode == "avg":
        maxdiff = 0.0
        for i, (s, _) in enumerate(pieces):
            sub = tok0(s, add_special_tokens=False)["input_ids"]
            maxdiff = max(maxdiff, float((new[i] - orig[sub].mean(0)).abs().max()))
        assert maxdiff < 1e-5, maxdiff
        msg = f"avg: max |row - mean(original sub-pieces)| = {maxdiff:.2e}"
    else:
        m, sd = float(new.mean()), float(new.std())
        assert abs(m) < 0.01 and abs(sd - 1.0) < 0.01, (m, sd)
        msg = f"rand: mean={m:.4f} std={sd:.4f} (expected N(0,1)); std of old rows={float(orig.std()):.3f}"
    print(f"\n[4] {mode}: old rows unchanged, <extra_id_0>=36095 keeps its vector, tied OK; {msg}; info={info}")


def test_5_save_reload(data, toks, tmp_path):
    from transformers import PreTrainedTokenizerFast
    d, df_w = data
    _, tokx, _, _ = toks
    tokx.save_pretrained(tmp_path)
    r1 = PreTrainedTokenizerFast.from_pretrained(tmp_path)
    r2 = R.load_t5_tokenizer_v4compat("VietAI/vit5-base", tokenizer_file=str(tmp_path / "tokenizer.json"))
    xs = d[R.COL_M].tolist() + d[R.COL_V].tolist()
    a = tokx(xs)["input_ids"]
    assert a == r1(xs)["input_ids"] == r2(xs)["input_ids"]
    assert tokx.batch_decode(a, skip_special_tokens=True) == r2.batch_decode(a, skip_special_tokens=True)
    print(f"\n[5] save/reload: identical encodings on {len(xs)} sentences")
