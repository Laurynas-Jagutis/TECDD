"""Run the ProDrift port on one log, for a grid of settings (tuning on the validation log) or a
single setting (the test run), and save the ranked scores to a CSV.

Every 5 events, the G-test (p <= 0.05) compares the alpha+ relations of two adjacent windows of
the complete events. Window size, noise filter and sensitivity are tuned. The p-value series
only depends on window size and noise filter, so every sensitivity is scored from the same
series, and each row's runtime is what an online run with that setting costs (p-values plus
drift retrieval).

Scoring is the same as for TECDD: drifts and alarms are positions in the full time-ordered log
(every lifecycle transition), and a detection d is a true positive if
start <= d <= end + 500 (end = start for sudden drifts), matched in order and one to one.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from time import perf_counter

from prodrift_events import compute_pvalues, load_event_stream, retrieve_drifts

STEP = 5
TOLERANCE = 500


def load_ground_truth(csv_path):
    """Drift start cases and, for gradual logs, the last old case of each transition (0-based)."""
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    starts = [int(row["drift_case_index_0based"]) for row in rows]
    ends = ([int(row["transition_end_case_index_0based"]) for row in rows]
            if rows and "transition_end_case_index_0based" in rows[0] else None)
    return starts, ends


def drift_positions(full_case_indices, start_cases, end_cases, case_count):
    """1-based full log positions of each drift start (first event of the new block's cases) and
    end (first event of the transition's last old case, = start for sudden drifts)."""
    first_event = {}
    for position, case in enumerate(full_case_indices, start=1):
        first_event.setdefault(case, position)
    starts = []
    for k, start_case in enumerate(start_cases):
        next_case = start_cases[k + 1] if k + 1 < len(start_cases) else case_count
        starts.append(next(position for position, case in enumerate(full_case_indices, start=1)
                           if start_case <= case < next_case))
    ends = [first_event[case] for case in end_cases] if end_cases is not None else starts
    return starts, ends


def score_detections(true_starts, true_ends, detections, tolerance=TOLERANCE):
    i = j = tp = fp = fn = 0
    delays = []
    while i < len(true_starts) and j < len(detections):
        if true_starts[i] <= detections[j] <= max(true_starts[i], true_ends[i]) + tolerance:
            tp += 1
            delays.append(detections[j] - true_starts[i])
            i += 1
            j += 1
        elif detections[j] < true_starts[i]:
            fp += 1
            j += 1
        else:
            fn += 1
            i += 1
    fn += len(true_starts) - i
    fp += len(detections) - j
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "mean_delay_events": sum(delays) / len(delays) if delays else None}


def rank_key(row):
    """Best F1 first, then the lower delay, then the faster run."""
    return (-row["f1"], math.inf if row["mean_delay_events"] is None else row["mean_delay_events"],
            row["runtime_seconds"])


def write_ranked(output_csv, rows):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=rank_key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Tune or test the ProDrift port on one log.")
    parser.add_argument("--xes-path", type=Path, required=True)
    parser.add_argument("--drift-ground-truth-csv", type=Path, required=True)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[100, 250, 500, 1000, 1500, 2000],
                        help="Events in each of the two windows.")
    parser.add_argument("--noise-filters", type=float, nargs="+", default=[5.0, 10.0], help="Relation noise filter (%%).")
    parser.add_argument("--sensitivities", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Persistence = int(window * (1 - sensitivity)) events.")
    parser.add_argument("--max-runtime-hours", type=float, default=None,
                        help="Time budget (4 for BPIC 2012, 5 for BPIC 2015).")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    load_start = perf_counter()
    stream, case_count, full_case_indices = load_event_stream(args.xes_path)
    print(f"{len(stream):,} complete events of {len(full_case_indices):,}, {case_count:,} cases "
          f"(loaded in {perf_counter() - load_start:.1f}s)", flush=True)
    start_cases, end_cases = load_ground_truth(args.drift_ground_truth_csv)
    true_starts, true_ends = drift_positions(full_case_indices, start_cases, end_cases, case_count)

    combinations = list(itertools.product(args.window_sizes, args.noise_filters))
    budget_seconds = args.max_runtime_hours * 3600.0 if args.max_runtime_hours is not None else None
    rows = []
    grid_start = perf_counter()
    for index, (window_size, noise) in enumerate(combinations, start=1):
        if rows and budget_seconds is not None and perf_counter() - grid_start >= budget_seconds:
            print("Time budget reached, stopping.", flush=True)
            break
        print(f"[{index}/{len(combinations)}] window={window_size} noise={noise}%", flush=True)
        if len(stream) < 2 * window_size:
            print("    skipped: log shorter than two windows", flush=True)
            continue

        pvalue_start = perf_counter()
        pvalues = compute_pvalues(stream, window_size, noise_filter_percentage=noise, step=STEP)
        pvalue_seconds = perf_counter() - pvalue_start

        for sensitivity in args.sensitivities:
            retrieve_start = perf_counter()
            detections = retrieve_drifts(pvalues, window_size, sensitivity=sensitivity, step=STEP)
            runtime_seconds = pvalue_seconds + (perf_counter() - retrieve_start)
            predicted = [stream[position].full_position for position in detections]
            metrics = score_detections(true_starts, true_ends, predicted)
            rows.append({
                "window_size": window_size,
                "noise_filter": noise,
                "sensitivity": sensitivity,
                **metrics,
                "detections": len(predicted),
                "true_drift_count": len(true_starts),
                "runtime_seconds": runtime_seconds,
                "runtime_per_event": runtime_seconds / len(stream),
                "true_drifts": json.dumps(true_starts),
                "predicted_drifts": json.dumps(predicted),
            })
            print(f"    sensitivity={sensitivity}: f1={metrics['f1']:.3f} detections={len(predicted)} "
                  f"delay={metrics['mean_delay_events']}", flush=True)
        write_ranked(args.output_csv, rows)  # keep finished rows on disk

    if not rows:
        print("No setting could be run.", flush=True)
        return
    best = write_ranked(args.output_csv, rows)[0]
    print(f"\nBest: window={best['window_size']} noise={best['noise_filter']} sensitivity={best['sensitivity']} "
          f"-> f1={best['f1']:.3f}. Saved {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
