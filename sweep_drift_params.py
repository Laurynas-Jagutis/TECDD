"""Tune W, s, m and B of TECDD on a validation log by F1, within a time budget.

Finished combinations are saved after every run, so an interrupted sweep resumes where it
stopped when started again with the same --output-csv.
"""
import argparse
import csv
import io
import itertools
import json
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from main_drift import run_pipeline

HYPERPARAM_KEYS = ["window_size", "stride", "persistence", "bank_size"]
RESULT_KEYS = HYPERPARAM_KEYS + [
    "rank", "tp", "fp", "fn", "precision", "recall", "f1", "mean_delay_events",
    "detections", "true_drift_count", "runtime_seconds", "runtime_per_event",
]


def build_combinations(args):
    return [
        {"window_size": w, "stride": s, "persistence": m, "bank_size": b}
        for w, s, b, m in itertools.product(args.window_size, args.stride, args.bank_size, args.persistence)
    ]


def combo_key(row):
    return tuple(int(float(row[key])) for key in HYPERPARAM_KEYS)


def combo_summary(combo):
    return f"W={combo['window_size']} s={combo['stride']} m={combo['persistence']} B={combo['bank_size']}"


def parse_csv_value(value):
    if value is None or value == "":
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def load_existing_results(output_csv, combinations):
    """Rows of an earlier, interrupted run of this sweep. Refuses rows outside the current grid."""
    if not output_csv.is_file():
        return []
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = [{key: parse_csv_value(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    grid_keys = {combo_key(combo) for combo in combinations}
    if any(combo_key(row) not in grid_keys for row in rows):
        raise ValueError(f"Cannot resume from {output_csv}: it has rows outside the current grid.")
    return rows


def rank_results(results):
    results.sort(
        key=lambda row: (
            -row["f1"],
            -row["precision"],
            abs(row["detections"] - row["true_drift_count"]),
            row["window_size"],
            row["stride"],
        )
    )
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank
    return results


def save_progress(output_csv, results):
    """Rank and write everything finished so far, plus the best configuration."""
    rank_results(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp_path = output_csv.with_name(output_csv.name + ".tmp")
        with tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_KEYS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        os.replace(tmp_path, output_csv)
        with (output_csv.parent / "best_config.json").open("w", encoding="utf-8") as handle:
            json.dump({key: results[0][key] for key in RESULT_KEYS}, handle, indent=2)
    except PermissionError as exc:
        print(f"    WARNING: could not save progress ({exc}); will retry after the next combination.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Tune W, s, m and B of TECDD on a validation log.")
    parser.add_argument("--xes-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--drift-ground-truth-csv", type=Path, required=True)
    parser.add_argument("--window-size", type=int, nargs="+", default=[150, 200, 250], help="W grid")
    parser.add_argument("--stride", type=int, nargs="+", default=[12, 18, 24], help="s grid")
    parser.add_argument("--persistence", type=int, nargs="+", default=[6, 8, 10], help="m grid")
    parser.add_argument("--bank-size", type=int, nargs="+", default=[10, 25, 50], help="B grid")
    parser.add_argument("--max-runtime-hours", type=float, default=None, help="Time budget (4 for BPIC 2012, 5 for BPIC 2015).")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    combinations = build_combinations(args)
    results = load_existing_results(args.output_csv, combinations)
    done_keys = {combo_key(row) for row in results}
    # Time already spent by an interrupted run counts towards the budget.
    offset_seconds = float(sum(row.get("runtime_seconds") or 0.0 for row in results))
    max_runtime_seconds = args.max_runtime_hours * 3600 if args.max_runtime_hours is not None else None
    print(f"Running {len(combinations)} combinations ({len(results)} already done)...", flush=True)

    start_time = time.monotonic()

    def budget_spent():
        return max_runtime_seconds is not None and (time.monotonic() - start_time + offset_seconds) >= max_runtime_seconds

    for idx, combo in enumerate(combinations, start=1):
        if combo_key(combo) in done_keys:
            continue
        if budget_spent():
            print("Time budget reached, stopping.", flush=True)
            break

        print(f"[{idx}/{len(combinations)}] {combo_summary(combo)}", flush=True)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                run_result = run_pipeline(
                    xes_path=args.xes_path,
                    checkpoint_path=args.checkpoint_path,
                    ground_truth_csv=args.drift_ground_truth_csv,
                    **combo,
                )
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            continue

        metrics = run_result["metrics"]
        results.append({
            **combo,
            **metrics,
            "detections": len(run_result["detected_drifts"]),
            "true_drift_count": run_result["true_drift_count"],
            "runtime_seconds": run_result["runtime_seconds"],
            "runtime_per_event": run_result["runtime_per_event"],
        })
        print(
            f"    f1={metrics['f1']:.3f} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"detections={len(run_result['detected_drifts'])} runtime={run_result['runtime_seconds']:.1f}s",
            flush=True,
        )
        save_progress(args.output_csv, results)

    if not results:
        print("No successful runs completed.", flush=True)
        return
    save_progress(args.output_csv, results)
    print(f"\nBest of {len(results)}: {combo_summary(results[0])} -> f1={results[0]['f1']:.3f}", flush=True)
    print(f"Saved results to {args.output_csv} and best_config.json next to it.", flush=True)


if __name__ == "__main__":
    main()
