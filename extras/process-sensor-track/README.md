# Optional extra track — defect detection from in-line process sensors (UCI SECOM)

**This is not part of the main project, and nothing in the main pipeline depends on
it. Keep it or delete the directory; either is fine.**

## Why it exists

It was built during a session in which the repository's own WM-811K project was not
yet visible to the agent, and the task was read as "find a public wafer dataset and
train something on it". The container's network policy blocked every host that serves
WM-811K and MixedWM38 (Kaggle, Google Drive, Hugging Face, Zenodo, Baidu, Tianchi,
mirlab), so the nearest reachable **real, public, citable semiconductor defect
dataset** was UCI SECOM — and that is what this track uses.

It is kept because it is complete, tested, and it answers a different question from
the main project: *can end-of-line pass/fail be predicted from in-line process sensor
readings?* That is a tabular, yield-prediction framing, complementary to the
wafer-map imaging framing of the main tracks.

## What it is

UCI SECOM: 1,567 production runs × 590 in-line process sensors, labelled pass/fail
by end-of-line testing, timestamped over three months of 2008. 104 runs (6.6%) fail.

| piece | what it does |
|---|---|
| `src/sicdd/data.py` | loading, SHA-256 provenance checks, chronological split |
| `src/sicdd/preprocess.py` | missing-rate filter → median impute → constant-column drop → scaling, all fitted on training runs only |
| `src/sicdd/models.py` | prior baseline, logistic regression, random forest, hist-gradient boosting, and a class-weighted PyTorch MLP |
| `src/sicdd/evaluate.py` | average precision, cost-sensitive thresholds, inspection-budget recall, bootstrap CIs |
| `src/sicdd/train.py` | the experiment runner (see protocol below) |
| `src/sicdd/predict.py` | applies a saved bundle to new process data |
| `src/sicdd/figures.py` | the figures in `reported/figures/` |

## Protocol

1. Runs are time-ordered; the last 20% are locked away. No imputation statistic,
   hyperparameter or threshold is fitted on them.
2. Hyperparameters are selected by forward-chaining CV (`TimeSeriesSplit`), which
   only validates on runs later than the ones it trained on.
3. The same configuration is **also** scored under random stratified k-fold CV. The
   gap measures how much a random split flatters a model on a drifting process.
4. Operating points are fixed without touching the test runs, under two policies:
   minimum expected cost, and a fixed inspection budget. Thresholds transfer as
   *flag rates* mapped onto the refitted model's own score distribution, because
   partial-fold models emit a different score scale.

## What it found

Recorded in `reported/metrics.json` and `reported/leaderboard.csv`:

- **The failure rate drifts hard** — 22% → 9% → 2.9% → 6.1% by month. Every model
  shows a positive optimism gap: random 5-fold CV overstates average precision by
  0.014–0.066 relative to forward-chaining CV.
- **Model ranking is not identifiable at this sample size.** Fold-to-fold average
  precision swings enormously (random forest: 0.054 → 0.470 across folds), the
  rank correlation between CV and held-out performance is 0.5 over five models, and
  all bootstrap 95% CIs overlap. The protocol selects random forest on CV; on the
  future window logistic regression is the only model whose AP interval excludes
  the no-skill rate (0.101–0.420 vs. a 0.054 prevalence). **These were not
  re-ranked after seeing the test set**, and a bundle is shipped for every model
  rather than pinning the deliverable to one.
- **Cost-optimal thresholding abstains.** At a 10:1 miss-to-false-alarm ratio,
  "flag nothing" is genuinely cost-optimal on held-out validation data, even though
  flagging the top 20% would have saved 13.5% on the test window. Reported as a
  negative result rather than tuned away.

Honest summary: this is a weak-signal dataset and the main value here is the
protocol and the negative results, not the scores.

## Reproducing it

The data is not committed (`data/` is gitignored, and these are not our files to
redistribute). Download `secom.data` and `secom_labels.data` from the UCI Machine
Learning Repository's SECOM page and put them in `data/raw/`. The loader verifies
their SHA-256, so a wrong or modified copy fails loudly:

```
secom.data         4503bdb42e63687d1c7586bdc1796df9a7cb5ae9cc9c0443d84957a446de3aaf
secom_labels.data  2fb2e8b5dfe77ba96ecf0960ff00f2ba9311795383a8342d4588a7cb37c816ef
```

Use the **original UCI files**, not the widely circulated Kaggle CSV re-export: that
copy parses the DD/MM/YYYY timestamps ambiguously and corrupts 582 of them
(`01/08/2008` becomes 8 January instead of 1 August), destroying the run ordering
this protocol depends on. The feature matrices are otherwise bit-identical — we
checked, cell by cell.

```bash
pip install -e .            # from this directory
sicdd-train                 # full experiment; writes models/ and results/
sicdd-figures               # figures
pytest tests -q             # 53 tests

sicdd-predict --model models/sicdd_defect_detector.joblib --input new_runs.csv --output scored.csv
sicdd-predict --model models/sicdd_defect_detector.joblib --describe
```

Each saved bundle carries its own fitted preprocessing, threshold, feature schema and
provenance, so scoring new runs needs nothing from the training code.
