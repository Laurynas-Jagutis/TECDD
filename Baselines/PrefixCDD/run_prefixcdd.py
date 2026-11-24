"""Run PrefixCDD (its original code, BPIC_CDD.run_prefix_tree_detector) on one log, for a grid of
settings (tuning on the validation log) or a single setting (the test run), and save the ranked
scores to a CSV.

Window size, tree size and decay are tuned, the noise filter stays at 1. Scoring is the same as
for TECDD: drifts and alarms are positions in the full time-ordered log, and a detection d is a
true positive if start <= d <= end + 500 (end = start for sudden drifts), matched in order and
one to one. For sudden logs this is exactly PrefixCDD's own ordered_match.
"""
import argparse
import csv
import io
import itertools
import json
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter

NOISE = 1
TOLERANCE = 500


def load_ground_truth(csv_path):
    """Drift start cases (0-based) and, for gradual logs, {drift start event: transition end
    event} as 1-based full log positions."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    start_cases = [int(row["drift_case_index_0based"]) for row in rows]
    end_by_start = (
        {int(row["drift_event_index_0based"]) + 1: int(row["transition_end_event_index_0based"]) + 1 for row in rows}
        if rows and "transition_end_event_index_0based" in rows[0] else None
    )
    return start_cases, end_by_start


def score_detections(true_starts, true_ends, detections, tolerance=TOLERANCE):
    i = j = tp = fp = fn = 0
    delays = []
    while i < len(true_starts) and j < len(detections):
        if detections[j] < true_starts[i]:
            fp += 1
            j += 1
        elif detections[j] <= max(true_starts[i], true_ends[i]) + tolerance:
            tp += 1
            delays.append(detections[j] - true_starts[i])
            i += 1
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


def run_once(xes_path, start_cases, end_by_start, window_size, tree_size, decay_lambda):
    from BPIC_CDD import run_prefix_tree_detector

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = run_prefix_tree_detector(
            file_path=xes_path, window_size=window_size, tree_size=tree_size, decay_lambda=decay_lambda,
            noise=NOISE, drift_trace_indices=start_cases, metric_tolerances=[TOLERANCE],
        )
    true_starts = result["true_global_drifts"]
    if end_by_start is None:
        true_ends = true_starts
    else:
        missing = [start for start in true_starts if start not in end_by_start]
        if missing:
            raise ValueError(f"Drift starts {missing} are not in the ground truth, so the event axes disagree.")
        true_ends = [end_by_start[start] for start in true_starts]
    predicted = result["detected_global_drifts"]
    return {
        "window_size": window_size,
        "tree_size": tree_size,
        "decay_lambda": decay_lambda,
        **score_detections(true_starts, true_ends, predicted),
        "detections": len(predicted),
        "true_drift_count": len(true_starts),
        "runtime_seconds": result["runtime_seconds"],
        "runtime_per_event": result["runtime_ms_per_event"] / 1000.0,
        "true_drifts": json.dumps(true_starts),
        "predicted_drifts": json.dumps(predicted),
    }


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
    parser = argparse.ArgumentParser(description="Tune or test PrefixCDD on one log.")
    parser.add_argument("--xes-path", type=Path, required=True)
    parser.add_argument("--drift-ground-truth-csv", type=Path, required=True)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[6, 9, 12, 15, 18],
                        help="Maximum number of prefix trees in the detection window.")
    parser.add_argument("--tree-sizes", type=int, nargs="+", default=[250, 500, 750, 1000, 1250],
                        help="Events per prefix tree (pruning steps).")
    parser.add_argument("--decay-lambdas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--max-runtime-hours", type=float, default=None,
                        help="Time budget (4 for BPIC 2012, 5 for BPIC 2015).")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    xes_path = args.xes_path.resolve()
    start_cases, end_by_start = load_ground_truth(args.drift_ground_truth_csv)
    combinations = list(itertools.product(args.window_sizes, args.tree_sizes, args.decay_lambdas))
    budget_seconds = args.max_runtime_hours * 3600.0 if args.max_runtime_hours is not None else None

    rows = []
    grid_start = perf_counter()
    for index, (window_size, tree_size, decay_lambda) in enumerate(combinations, start=1):
        if rows and budget_seconds is not None and perf_counter() - grid_start >= budget_seconds:
            print("Time budget reached, stopping.", flush=True)
            break
        print(f"[{index}/{len(combinations)}] window={window_size} tree={tree_size} decay={decay_lambda}", flush=True)
        try:
            row = run_once(xes_path, start_cases, end_by_start, window_size, tree_size, decay_lambda)
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            continue
        rows.append(row)
        print(f"    f1={row['f1']:.3f} detections={row['detections']} delay={row['mean_delay_events']} "
              f"runtime={row['runtime_seconds']:.1f}s", flush=True)
        write_ranked(args.output_csv, rows)  # keep finished rows on disk

    if not rows:
        print("No setting could be run.", flush=True)
        return
    best = write_ranked(args.output_csv, rows)[0]
    print(f"\nBest: window={best['window_size']} tree={best['tree_size']} decay={best['decay_lambda']} "
          f"-> f1={best['f1']:.3f}. Saved {len(rows)} rows to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
