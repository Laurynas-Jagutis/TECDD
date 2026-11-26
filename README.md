# TECDD

Code for **TECDD**, Transformer based Embeddings for unsupervised online Concept Drift Detection in event streams, together with the two baselines it is compared against (ProDrift and PrefixCDD) and the script that builds the 18 semi synthetic evaluation logs.

TECDD reads an event stream one event at a time. A transformer trained on next event prediction turns the prefix of each running case into an embedding. Windows of embeddings are compared to a reference bank with the maximum mean discrepancy (MMD), and a drift is reported when the distance stays above a calibrated threshold for several windows in a row. No drift labels are used for training, calibration or detection. They are only used to score the results.

## Repository layout

```text
.
├── README.md
├── requirements.txt               TECDD
├── train_next_event_model.py      TECDD stages 1 and 2: train the transformer, calibrate the detector
├── sweep_drift_params.py          TECDD: tune W, s, m and B on a validation log
├── run_repeated_test_config.py    TECDD: run one setting on a test log
├── main_drift.py                  TECDD stage 3: online monitoring and scoring
├── drift_detector.py, prefix_embedder.py, window_processor.py, case_buffer.py,
├── data_loader.py, dataset.py, event.py, time_preprocessing.py, trainer.py
├── models/sutran.py               the transformer
├── Baselines/
│   ├── ProDrift/                  Python port of ProDrift (event stream, fixed window mode)
│   │   ├── requirements.txt
│   │   ├── prodrift_events.py
│   │   └── run_prodrift.py
│   └── PrefixCDD/                 PrefixCDD by its original authors (two files adapted) plus our runner
│       ├── README.md              the authors' README
│       ├── requirements.txt
│       ├── requirements.yml       the authors' conda environment
│       ├── BPIC_CDD.py, CDD.py, DDScripts.py, HeuristicsAlgo.py, PrefixTreeClass.py, settings.py
│       └── run_prefixcdd.py
└── Dataset-Generation/
    ├── requirements.txt
    ├── build_semi_synthetic_logs.py   builds one semi synthetic log (validation and test split)
    ├── trace_features.py              per case features for clustering
    └── xes_io.py                      reading, copying and writing XES cases
```

## Environments

Two Python environments are needed. TECDD, ProDrift and the dataset builder share one. PrefixCDD needs its own, because it depends on old versions of pm4py and scikit-multiflow.

### TECDD, ProDrift and dataset generation (Python 3.13)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux or macOS
pip install -r requirements.txt -r Baselines/ProDrift/requirements.txt -r Dataset-Generation/requirements.txt
```

TECDD runs on the CPU, and uses a GPU automatically when PyTorch finds one. All reported results were produced on a CPU.

### PrefixCDD (Python 3.8)

The most reliable way is the authors' conda environment:

```bash
conda env create -f Baselines/PrefixCDD/requirements.yml
conda activate streamPM
```

Always start PrefixCDD from an activated conda environment (`conda activate streamPM`). Calling the environment's `python.exe` directly without activating it can fail with a DLL load error on Windows.

`Baselines/PrefixCDD/requirements.txt` lists only the packages the code imports, for a plain `pip install` into a Python 3.8 environment. The conda environment above is the one the reported results were produced with.

## 1. Building the semi synthetic logs

The 18 evaluation logs come from two base logs, BPI Challenge 2012 and BPI Challenge 2015, both available from 4TU.ResearchData. For BPIC 2015, the five municipality logs are merged into a single XES file first.

`Dataset-Generation/build_semi_synthetic_logs.py` builds one log per call. Each log is defined by a base log, a block order and a drift type:

| `--order` | Next block after each block | Difficulty in the paper |
|---|---|---|
| `farthest` | the unused block farthest from it | easy |
| `median` | the unused block at median distance | medium |
| `nearest` | the unused block nearest to it | hard |

| `--drift-type` | When the cases of a new block start |
|---|---|
| `simultaneous` | all at the block boundary |
| `staggered` | with the start gaps they had in the base log |
| `gradual` | interleaved with the previous block over a transition, the share of new cases rising linearly |

Example, the BPIC 2015 log with median order and gradual drifts:

```bash
python Dataset-Generation/build_semi_synthetic_logs.py --input-xes data/BPIC15_merged.xes --order median --drift-type gradual --output-dir data/2015_median_gradual
```

The 18 logs are the 2 base logs × 3 orders × 3 drift types. One call takes less than a minute.

### How a log is built

1. **Blocks.** Every case of the base log gets a feature vector: length, distinct activities and resources, duration, time gaps between events, the number of cases open at the same time, start time, the relative frequencies of the 40 most frequent activities, 80 most frequent directly follows pairs and 40 most frequent resources, and which activity and resource occur at each of the first 40 positions. The standardized features are reduced with PCA to 12 dimensions and clustered with k-means (k = 25, 50 restarts). Clusters closer than 0.5% of the largest distance between two clusters are merged, and clusters with fewer than 50 cases or 500 events are not used. Every remaining cluster is a block.
2. **Block order.** The sequence starts with the earliest block. After each block, the next one is chosen among the unused blocks by the distance between the blocks' mean feature vectors, as set by `--order`. At most 25 blocks are used (24 drifts).
3. **Timing.** The cases are copied with new case IDs, and all their timestamps are shifted, starting on 1 January 2026. The time between two blocks is the median gap between consecutive case starts in the base log. For gradual drifts, each transition holds about 2,000 events (at most 40% of the smaller block), taken from the end of the old block and the start of the new one.
4. **Split.** The log is split by blocks into a validation log with the first half of the drifts and a test log with the rest. The drift at the split point belongs to neither.

Everything is seeded, so the same base log always gives exactly the same logs. Every case in the output keeps its original attributes and gets three more: `synthetic:variant_id` (its block), `synthetic:block_index` (the block's position in the sequence) and `synthetic:original_case_id`.

Output:

```text
data/2015_median_gradual/
├── metadata.json                  block order, split and the gap between blocks
├── validation/
│   ├── event_log.xes
│   └── drift_ground_truth.csv
└── test/
    ├── event_log.xes
    └── drift_ground_truth.csv
```

`drift_ground_truth.csv` has one row per drift: `drift_id`, `from_variant_id`, `to_variant_id`, `drift_timestamp`, `drift_case_index_0based` and `drift_event_index_0based`. A drift starts at the first case of the new block, and the indexes are its 0-based case and event positions in the time ordered log. Gradual logs also have `transition_end_timestamp`, `transition_end_case_index_0based` and `transition_end_event_index_0based`, the position of the last case of the old block. Their presence switches every runner to gradual scoring (see [Scoring](#scoring)).

## 2. TECDD

Every log is processed in three steps. The validation log is used for training, calibration and tuning, and the test log only for the final run.

### Train and calibrate

```bash
python train_next_event_model.py --xes-path data/2015_median_gradual/validation/event_log.xes --checkpoint-path checkpoints/2015_median_gradual.pt
```

The transformer is trained for 20 epochs on next event prediction over all traces of the validation log (a random 20% of the traces selects the best epoch). The detector is then calibrated on the first 300 traces: PCA (95% of the variance), the reference sample R0, the kernel bandwidth h and the thresholds θ and θ′. Everything is stored in the checkpoint, and the threshold stays frozen afterwards. `--seed` (default 42) fixes the random state.

### Tune on the validation log

```bash
python sweep_drift_params.py --xes-path data/2015_median_gradual/validation/event_log.xes --checkpoint-path checkpoints/2015_median_gradual.pt --drift-ground-truth-csv data/2015_median_gradual/validation/drift_ground_truth.csv --max-runtime-hours 5 --output-csv results/tecdd/2015_median_gradual_sweep.csv
```

| Flag | Meaning | Default grid |
|---|---|---|
| `--window-size` | W, events per window | 150 200 250 |
| `--stride` | s, events between two windows | 12 18 24 |
| `--persistence` | m, windows in a row above the threshold before a drift is reported | 6 8 10 |
| `--bank-size` | B, windows in the reference bank | 10 25 50 |
| `--max-runtime-hours` | time budget, 4 for BPIC 2012 and 5 for BPIC 2015 | no limit |

Rows are ranked by F1, then precision, then how close the number of detections is to the number of true drifts. The CSV is rewritten after every combination, so a sweep that was interrupted continues where it stopped when it is started again with the same `--output-csv`.

### Run on the test log

Take W, s, m and B from the first row of the sweep CSV:

```bash
python run_repeated_test_config.py --xes-path data/2015_median_gradual/test/event_log.xes --checkpoint-path checkpoints/2015_median_gradual.pt --drift-ground-truth-csv data/2015_median_gradual/test/drift_ground_truth.csv --window-size 200 --stride 18 --persistence 8 --bank-size 25 --output-csv results/tecdd/2015_median_gradual_test.csv
```

The values above are only an example. The output has one row with the scores, the detected and true drift positions, the event at which monitoring started (`activation_event_index`) and the runtime.

### Fixed settings

The settings that are not tuned are constants at the top of the scripts:

- `train_next_event_model.py`: prefix length 10, 20 epochs, batch size 128, learning rate 1e-3, weight decay 1e-4, 300 calibration traces, calibration windows with W = 200 and s = 18.
- `main_drift.py`: cooldown of 100 windows after a drift, 250 warmup events, cold start bank of 50 windows sampled every third window, case eviction after 25,000 idle events, scoring tolerance of 500 events.

## 3. ProDrift

`prodrift_events.py` is a Python port of the event stream mode of ProDrift (Ostovar et al., 2016) with fixed windows and the default Alpha+ relations, following the Apromore Java implementation. Every 5 events, a G-test (p ≤ 0.05) compares the relations of two adjacent windows of complete events.

Tune on the validation log:

```bash
python Baselines/ProDrift/run_prodrift.py --xes-path data/2015_median_gradual/validation/event_log.xes --drift-ground-truth-csv data/2015_median_gradual/validation/drift_ground_truth.csv --max-runtime-hours 5 --output-csv results/prodrift/2015_median_gradual_sweep.csv
```

| Flag | Meaning | Default grid |
|---|---|---|
| `--window-sizes` | events in each of the two windows | 100 250 500 1000 1500 2000 |
| `--noise-filters` | relation noise filter in % | 5 10 |
| `--sensitivities` | the drift must persist for int(window × (1 − sensitivity)) events | 0.1 0.3 0.5 0.7 0.9 |
| `--max-runtime-hours` | time budget | no limit |

Run the best setting (the first row of the sweep CSV) on the test log by passing a single value to each grid flag:

```bash
python Baselines/ProDrift/run_prodrift.py --xes-path data/2015_median_gradual/test/event_log.xes --drift-ground-truth-csv data/2015_median_gradual/test/drift_ground_truth.csv --window-sizes 1000 --noise-filters 10 --sensitivities 0.5 --output-csv results/prodrift/2015_median_gradual_test.csv
```

## 4. PrefixCDD

The PrefixCDD files are the code of its original authors, with two files adapted for this evaluation. The detection itself (prefix tree construction, pruning and the drift test in the window) is unchanged.

- `BPIC_CDD.py`: the body of the original `main` is wrapped in a function, `run_prefix_tree_detector`, that returns the detections, the true drift positions and the runtime instead of only printing them. It also prints the number of events processed before the first drift check. The command line keeps the original six options and adds `--trace-changepoints`, `--case-changepoints`, `--metric-tolerances` and `--output-json`.
- `CDD.py`: a detected drift can be exported as a dictionary (`to_dict`).

`run_prefixcdd.py` calls `run_prefix_tree_detector` and scores the detections in the same way as for the other methods. Activate the conda environment first.

Tune on the validation log:

```bash
conda activate streamPM
python Baselines/PrefixCDD/run_prefixcdd.py --xes-path data/2015_median_gradual/validation/event_log.xes --drift-ground-truth-csv data/2015_median_gradual/validation/drift_ground_truth.csv --max-runtime-hours 5 --output-csv results/prefixcdd/2015_median_gradual_sweep.csv
```

| Flag | Meaning | Default grid |
|---|---|---|
| `--window-sizes` | maximum number of prefix trees in the detection window | 6 9 12 15 18 |
| `--tree-sizes` | events per prefix tree | 250 500 750 1000 1250 |
| `--decay-lambdas` | decay applied to older traces at each pruning step | 0.1 0.3 0.5 |
| `--max-runtime-hours` | time budget | no limit |

The noise filter is fixed at 1. Run the best setting on the test log:

```bash
python Baselines/PrefixCDD/run_prefixcdd.py --xes-path data/2015_median_gradual/test/event_log.xes --drift-ground-truth-csv data/2015_median_gradual/test/drift_ground_truth.csv --window-sizes 9 --tree-sizes 250 --decay-lambdas 0.1 --output-csv results/prefixcdd/2015_median_gradual_test.csv
```

Both baseline runners rank rows by F1, then mean delay, then runtime, and rewrite the CSV after every finished setting.

## Scoring

All methods are scored in the same way:

- Drifts and detections are positions in the full, time ordered event log.
- A sudden drift (simultaneous or staggered log) starts and ends at the first event of the new block. A gradual drift starts at the first event of the new block and ends where the transition ends.
- A detection d is a true positive for a drift if start ≤ d ≤ end + 500. True drifts and detections are matched in order, one to one. Unmatched detections are false positives, unmatched drifts are false negatives.
- The delay of a true positive is d − start, in events.

Each output CSV contains TP, FP, FN, precision, recall, F1, the mean delay, the number of detections, the runtime in seconds and per event, and the true and detected drift positions as JSON lists.
