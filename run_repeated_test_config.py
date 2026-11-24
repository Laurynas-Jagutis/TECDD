"""Run TECDD once on a test log with a fixed W, s, m and B, and save the scores to a CSV."""
import argparse
import csv
import json
from pathlib import Path

from main_drift import run_pipeline

FIELDNAMES = [
    "xes_path",
    "checkpoint_path",
    "window_size",
    "stride",
    "persistence",
    "bank_size",
    "threshold",
    "activation_event_index",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "mean_delay_events",
    "runtime_seconds",
    "runtime_per_event",
    "monitored_event_count",
    "detections",
    "true_drift_count",
    "true_drifts",
    "predicted_drifts",
    "training_runtime_seconds",
]


def main():
    parser = argparse.ArgumentParser(description="Run TECDD on a test log with a fixed configuration.")
    parser.add_argument("--xes-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--drift-ground-truth-csv", type=Path, required=True)
    parser.add_argument("--window-size", type=int, required=True, help="W")
    parser.add_argument("--stride", type=int, required=True, help="s")
    parser.add_argument("--persistence", type=int, required=True, help="m")
    parser.add_argument("--bank-size", type=int, required=True, help="B")
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    result = run_pipeline(
        xes_path=args.xes_path,
        checkpoint_path=args.checkpoint_path,
        ground_truth_csv=args.drift_ground_truth_csv,
        window_size=args.window_size,
        stride=args.stride,
        persistence=args.persistence,
        bank_size=args.bank_size,
    )
    metrics = result["metrics"]
    row = {
        "xes_path": str(args.xes_path),
        "checkpoint_path": str(args.checkpoint_path),
        "window_size": args.window_size,
        "stride": args.stride,
        "persistence": args.persistence,
        "bank_size": args.bank_size,
        "threshold": result["threshold"],
        "activation_event_index": result["activation_event_index"],
        **metrics,
        "runtime_seconds": result["runtime_seconds"],
        "runtime_per_event": result["runtime_per_event"],
        "monitored_event_count": result["monitored_event_count"],
        "detections": len(result["detected_drifts"]),
        "true_drift_count": result["true_drift_count"],
        # Full stream positions (0-based true drifts, 1-based alarm events), as for the baselines.
        "true_drifts": json.dumps(result["true_full_stream_drifts"], separators=(",", ":")),
        "predicted_drifts": json.dumps(
            [d["confirmation_global_event_idx"] for d in result["detected_drifts"]
             if d.get("confirmation_global_event_idx") is not None],
            separators=(",", ":"),
        ),
        "training_runtime_seconds": result["training_runtime_seconds"],
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved results to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
