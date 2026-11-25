# Results of the paper's runs

Aggregated outputs of `python run_vimu_experiments.py --report_only` over all runs of
the paper (no source, reference or predicted sentences).

| File | Content |
|---|---|
| `summary_raw.csv` | one row per run: BLEU, chrF++, TER, tone accuracy, copy rate, overlap bands, latency, ... |
| `summary_mean_std.csv` | mean and standard deviation over seeds per configuration (Tables II and III) |
| `significance.csv` | paired bootstrap tests (1,000 resamples) |
| `table_main.tex` | LaTeX version of the main table |
| `corpus_analysis.json` | corpus statistics (Table I) |

Every Table II cell of the paper (BLEU, standard deviation, chrF++, TER, tone accuracy
and latency) matches `summary_mean_std.csv` after rounding.
