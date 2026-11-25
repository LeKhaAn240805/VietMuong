# VietMuong: Vietnamese–Muong Machine Translation

Code for the paper

> **VietMuong: A Bilingual Corpus and Translation Benchmark for Vietnamese–Muong Machine Translation**
> An Le Kha, Duong Tran Ham and Tri Pham Xuan. *RIVF 2026.*

- **Dataset:** <https://huggingface.co/datasets/LKAusagi/VietMuong> (15,377 lexical pairs, 12,839 sentence pairs)
- **Code:** one script, [`run_vimu_experiments.py`](run_vimu_experiments.py), runs the whole protocol of the paper: data preparation, baselines, fine-tuning, evaluation, significance tests and the result tables.
- **Results:** the aggregated numbers behind the paper's tables are in [`results/`](results/).

## Contents

```
run_vimu_experiments.py   experiment runner (data, training, evaluation, report)
tests/                    data-pipeline and vocabulary-expansion checks (no GPU)
results/                  aggregated results of the paper's runs (no text data)
scripts/                  reproduction commands
DATA_NOTICE.md            data source, copyright and third-party licenses
```

## Data

The dataset is **not** stored in this repository. The script downloads it from the
Hugging Face Hub at the revision used for the paper
(`d129079b0d8bf2c887f41ad879d884d9f3ad15c0`):

```python
from datasets import load_dataset

sentences = load_dataset("LKAusagi/VietMuong", "sentence_pairs")   # SentenceM, SentenceV
words = load_dataset("LKAusagi/VietMuong", "word_pairs")            # WordM, WordV
```

Local CSV files with the same columns can be used instead with
`--sentence_file` and `--word_file`.

The data were extracted from a copyrighted printed dictionary. Read
[DATA_NOTICE.md](DATA_NOTICE.md) before using or redistributing them.

## Installation

Tested with Python 3.10, CUDA 13.0 and NVIDIA RTX A5000 (24 GB) GPUs.

```bash
conda create -n vietmuong python=3.10 -y
conda activate vietmuong
pip install -r requirements.txt      # install the torch build that matches your CUDA first if needed
```

Quick checks (no GPU needed):

```bash
python run_vimu_experiments.py --self_test
python -m pytest -q tests/
```

`tests/test_data.py` checks the numbers of Sec. III-D (12,839 pairs, 12,800 groups,
10,272 / 1,284 / 1,283 split for seeds 42–46) and the instance counts of Table III.
`tests/test_vocab_expansion.py` checks the vocabulary expansion (2,727 added items).

## Reproducing the paper

```bash
python run_vimu_experiments.py --dry_run --gpus 0,1      # print the plan and its cost
python run_vimu_experiments.py --gpus 0,1                # run everything
python run_vimu_experiments.py --report_only             # tables, tests and figures
```

Each GPU runs one job at a time from a shared queue. Finished runs are skipped and
unfinished runs resume from their last checkpoint, so the same command can be
repeated after an interruption. Single experiments can be run with `--experiments`
and single runs with `--only_run_ids` (see [scripts/reproduce_paper.sh](scripts/reproduce_paper.sh)).

| Paper | Experiment | Output (in `--out_dir`, default `./vimu_runs`) |
|---|---|---|
| Table I (corpus statistics) | every run | `corpus_analysis.json` |
| Table II (main results) | `E1_main` + CPU baselines | `summary_mean_std.csv`, `table_main.tex` |
| Table III (data-strategy ablation) | `E3_strategy` | `summary_mean_std.csv`, `significance.csv` |
| Sec. V-C (vocabulary expansion) | `E3_strategy` (`token`, `token_avg`) | `summary_mean_std.csv`, `significance.csv` |
| Table IV (surface-overlap bands, Δtone) | `E1_main`, `E3_strategy` | `summary_raw.csv` (`bleu_cognate_*`, `tone_induced_bleu_gap`) |
| Significance tests | all | `significance.csv` (paired bootstrap, 1,000 resamples) |

`E4_contraction` (ViT5-base at the 70:10:20 and 60:10:30 splits) also runs by
default; it is supplementary and not reported in the paper's tables. Leave it out
with `--experiments E1_main E3_strategy`.

### Protocol

| | |
|---|---|
| Split | exact deduplication, near-duplicate groups (Jaccard ≥ 0.8), group-level 80:10:10 split; lexical pairs only in training |
| Seeds | 42–44 for neural models, 42–46 for the copy and rule-based baselines |
| Training | Adafactor, effective batch size 16, BF16; learning rates and epoch caps per model in `MODEL_REGISTRY` |
| Model selection | early stopping (patience 3) on BLEU of a fixed subset of 500 validation pairs |
| Test | beam size 4; sacreBLEU 2.6.0 BLEU (13a, mixed case), chrF++ (char order 6, word order 2), TER, tone accuracy |
| Latency | seed 42, batch size 1, beam size 4, first 100 test sentences, measured inside the run after test decoding, one run per GPU |

All 162 neural runs (E1, E3 and the supplementary E4) took about 280 GPU-hours
on RTX A5000 GPUs; `--dry_run` prints an estimate for any selection. Latency depends on the hardware and on the load of the machine,
so the `ms` column will not be reproduced exactly on other systems.

### Software versions

[`requirements.txt`](requirements.txt) pins the versions the script was verified with
(Python 3.10, transformers 5.18, torch 2.14). With the same GPU type and software
versions, a run reproduces its BLEU and its predictions exactly.

### Notes on the paper text

- Near-duplicates are grouped with Jaccard similarity over **syllable** 3-gram
  shingles of the normalised Muong side (`dedup_and_group`). Sec. III-D says
  "character 3-gram shingles"; the code is what produced the reported 12,800 groups
  and splits.
- The Hugging Face release already contains the 12,839 deduplicated sentence pairs,
  so a fresh run reports `raw = 12,839`; Table I reports the 12,863 pairs before
  exact deduplication.
- The extraction pipeline of Sec. III-A (EasyOCR + Qwen3-VL-2B-Instruct and the
  manual verification) is not part of this release.

## Citation

```bibtex
@inproceedings{lekha2026vietmuong,
  title     = {VietMuong: A Bilingual Corpus and Translation Benchmark for Vietnamese--Muong Machine Translation},
  author    = {Le Kha, An and Tran Ham, Duong and Pham Xuan, Tri},
  booktitle = {Proceedings of the 2026 RIVF International Conference on Computing and Communication Technologies (RIVF)},
  year      = {2026}
}
```

Please also cite the source dictionary (see [DATA_NOTICE.md](DATA_NOTICE.md)).

## License

The code is released under the [MIT License](LICENSE). The dataset, the source
dictionary and the pre-trained models have their own terms; see
[DATA_NOTICE.md](DATA_NOTICE.md).

## Contact

An Le Kha — lekhaank16@siu.edu.vn
