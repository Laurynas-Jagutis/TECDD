import gzip
import json
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict, deque
from itertools import islice
from pathlib import Path
from time import perf_counter

import click as click
import numpy as np
import pandas as pd
from pm4py.objects.conversion.log import converter as stream_converter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.streaming.importer.xes import importer as stream_xes_importer
from skmultiflow.drift_detection import ADWIN, PageHinkley

import settings
from CDD import Window
from PrefixTreeClass import PrefixTree

DEFAULT_METRIC_TOLERANCES = (1000, 2000, 3000, 5000)
XES_NS = {"xes": "http://www.xes-standard.org/"}


def parse_int_csv(value):
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    return [int(part.strip()) for part in text.split(",") if part.strip()]


def get_all_drift_indices(xes_path):
    if str(xes_path).endswith(".xes.gz"):
        with gzip.open(xes_path, "rb") as handle:
            root = ET.parse(handle).getroot()
    else:
        root = ET.parse(xes_path).getroot()

    drift_indices = []
    drift_info = root.find('xes:string[@key="drift:info"]', XES_NS)
    if drift_info is None:
        return drift_indices

    def find_trace_idx_recursive(element):
        found_idx = element.find('xes:string[@key="change_trace_index"]', XES_NS)
        if found_idx is not None:
            return found_idx
        for child in element:
            res = find_trace_idx_recursive(child)
            if res is not None:
                return res
        return None

    for drift_node in drift_info:
        idx_node = find_trace_idx_recursive(drift_node)
        if idx_node is None:
            continue

        clean_val = idx_node.attrib["value"].replace("[", "").replace("]", "")
        try:
            drift_indices.append(int(clean_val))
        except ValueError:
            pass

    return sorted(drift_indices)


def compute_true_global_drift_indices(log, xes_path=None, drift_trace_indices=None):
    resolved_trace_drifts = (
        sorted(int(idx) for idx in drift_trace_indices)
        if drift_trace_indices is not None
        else get_all_drift_indices(xes_path)
    )
    if not resolved_trace_drifts:
        return [], []

    segment_ranges = []
    for idx, start_trace_idx in enumerate(resolved_trace_drifts):
        end_trace_idx = resolved_trace_drifts[idx + 1] if idx + 1 < len(resolved_trace_drifts) else len(log)
        segment_ranges.append((start_trace_idx, end_trace_idx))

    ordered_events = []
    for trace_idx, trace in enumerate(log):
        for event_idx, event in enumerate(trace):
            ordered_events.append((event["time:timestamp"], len(ordered_events), trace_idx, event_idx))

    sorted_events = sorted(ordered_events, key=lambda item: (item[0], item[1]))
    true_global_drifts = [
        next(
            global_event_index
            for global_event_index, (_, _, trace_idx, _) in enumerate(sorted_events, start=1)
            if start_trace_idx <= trace_idx < end_trace_idx
        )
        for start_trace_idx, end_trace_idx in segment_ranges
    ]
    return resolved_trace_drifts, true_global_drifts


def ordered_match(true_indices, detected_indices, tolerance):
    i = 0
    j = 0
    matches = []
    false_positives = []
    false_negatives = []

    while i < len(true_indices) and j < len(detected_indices):
        true_idx = true_indices[i]
        det_idx = detected_indices[j]

        if det_idx < true_idx:
            false_positives.append(det_idx)
            j += 1
        elif det_idx <= true_idx + tolerance:
            matches.append((true_idx, det_idx))
            i += 1
            j += 1
        else:
            false_negatives.append(true_idx)
            i += 1

    false_negatives.extend(true_indices[i:])
    false_positives.extend(detected_indices[j:])
    return matches, false_positives, false_negatives


def evaluate_detections(true_drifts, detected_drifts, tolerances):
    results = []
    for tolerance in sorted({int(t) for t in tolerances}):
        matches, false_positives, false_negatives = ordered_match(true_drifts, detected_drifts, tolerance)

        tp = len(matches)
        fp = len(false_positives)
        fn = len(false_negatives)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

        signed_latencies = [det_idx - true_idx for true_idx, det_idx in matches]
        abs_errors = [abs(err) for err in signed_latencies]
        detection_delays = [max(0, latency) for latency in signed_latencies]

        results.append(
            {
                "tolerance": tolerance,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "matches": matches,
                "mean_signed_latency": float(np.mean(signed_latencies)) if signed_latencies else None,
                "mean_detection_delay": float(np.mean(detection_delays)) if detection_delays else None,
                "median_detection_delay": float(np.median(detection_delays)) if detection_delays else None,
                "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
                "median_abs_error": float(np.median(abs_errors)) if abs_errors else None,
            }
        )

    return results


def get_metric(metrics, tolerance):
    for stats in metrics:
        if stats["tolerance"] == tolerance:
            return stats
    raise ValueError(f"No metric found for tolerance {tolerance}.")


def run_prefix_tree_detector(
    file_path,
    window_size=10,
    tree_size=1000,
    decay_lambda=0.25,
    noise=1,
    configuration=False,
    drift_trace_indices=None,
    metric_tolerances=DEFAULT_METRIC_TOLERANCES,
):
    metric_tolerances = tuple(metric_tolerances) if metric_tolerances else DEFAULT_METRIC_TOLERANCES
    resolved_file_path = os.path.abspath(file_path)
    overall_start = perf_counter()

    settings.init()
    endEventsDic = dict()
    window = Window(initWinSize=window_size)

    streaming_ev_object = stream_xes_importer.apply(
        resolved_file_path,
        variant=stream_xes_importer.Variants.XES_TRACE_STREAM,
    )

    for trace in streaming_ev_object:
        lastEvent = trace[-1]["concept:name"]
        timeStamp = trace[-1]["time:timestamp"]
        caseID = trace.attributes["concept:name"]
        endEventsDic[caseID] = [lastEvent, timeStamp]

    caseList = []
    Dcase = OrderedDict()

    tree = PrefixTree(pruningSteps=tree_size, noiseFilter=noise, lambdaDecay=decay_lambda)
    adwin = ADWIN()
    ph = PageHinkley()

    pruningCounter = 0
    traceCounter = 0

    variant = xes_importer.Variants.ITERPARSE
    parameters = {
        variant.value.Parameters.TIMESTAMP_SORT: True,
        variant.value.Parameters.TIMESTAMP_KEY: "time:timestamp",
    }
    log = xes_importer.apply(resolved_file_path, variant=variant)

    static_event_stream = stream_converter.apply(
        log,
        variant=stream_converter.Variants.TO_EVENT_STREAM,
        parameters=parameters,
    )
    static_event_stream._list.sort(key=lambda x: x["time:timestamp"], reverse=False)

    eventCounter = 0
    first_cdd_event_count = None
    currentNode = tree.root
    processing_start = perf_counter()

    for ev in static_event_stream:
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(
            caseList,
            Dcase,
            currentNode,
            ev,
            pruningCounter,
            traceCounter,
            endEventsDic,
            window,
        )
        eventCounter += 1

        if window.cddFlag:
            if len(window.prefixTreeList) == window.WinSize:
                if first_cdd_event_count is None:
                    first_cdd_event_count = eventCounter
                    print(
                        f"[Cold-start] PrefixCDD: {first_cdd_event_count} events processed before the "
                        f"first drift check (reference window of {window_size} trees x {tree_size} events).",
                        flush=True,
                    )
                window.conceptDriftDetection(adwin, ph)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)

                if len(window.prefixTreeList) == window.WinSize:
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))

    processing_runtime_seconds = perf_counter() - processing_start
    runtime_seconds = perf_counter() - overall_start

    resolved_trace_drifts, true_global_drifts = compute_true_global_drift_indices(
        log,
        xes_path=resolved_file_path,
        drift_trace_indices=drift_trace_indices,
    )
    detected_global_drifts = [drift.eventsSeen for drift in window.driftsIdentified]
    metrics = (
        evaluate_detections(true_global_drifts, detected_global_drifts, tolerances=metric_tolerances)
        if true_global_drifts
        else []
    )

    result = {
        "file_path": resolved_file_path,
        "runtime_seconds": runtime_seconds,
        "processing_runtime_seconds": processing_runtime_seconds,
        "num_events": eventCounter,
        "num_traces": len(log),
        "cold_start_events": first_cdd_event_count,
        "runtime_ms_per_event": ((runtime_seconds * 1000.0) / eventCounter) if eventCounter > 0 else None,
        "processing_runtime_ms_per_event": (
            (processing_runtime_seconds * 1000.0) / eventCounter
        )
        if eventCounter > 0
        else None,
        "events_per_second": (eventCounter / runtime_seconds) if runtime_seconds > 0 else None,
        "traces_per_second": (len(log) / runtime_seconds) if runtime_seconds > 0 else None,
        "drift_trace_indices": resolved_trace_drifts,
        "true_global_drifts": true_global_drifts,
        "detected_global_drifts": detected_global_drifts,
        "detected_drifts": [drift.to_dict() for drift in window.driftsIdentified],
        "metrics": metrics,
    }

    if configuration:
        logProvider(static_event_stream, window)

    return result


def print_evaluation_summary(result):
    print("=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Log: {result['file_path']}")
    print(f"Runtime (end-to-end): {result['runtime_seconds']:.3f}s")
    print(f"Runtime (stream processing only): {result['processing_runtime_seconds']:.3f}s")
    if result["events_per_second"] is not None and result["traces_per_second"] is not None:
        print(
            f"Throughput: {result['events_per_second']:.2f} events/s | "
            f"{result['traces_per_second']:.2f} traces/s"
        )
    print(f"Events processed: {result['num_events']}")
    print(f"Traces processed: {result['num_traces']}")
    print(f"Ground Truth Trace Indices: {result['drift_trace_indices']}")
    print(f"Ground Truth Global Event Indices: {result['true_global_drifts']}")
    print(f"Detected Global Event Indices: {result['detected_global_drifts']}")

    if not result["true_global_drifts"]:
        print("No ground-truth drift metadata supplied or found, skipping F1/latency metrics.")
        return

    print("\nEvaluation Metrics:")
    for stats in result["metrics"]:
        print(
            f"Lag window 0..+{stats['tolerance']} events -> "
            f"TP: {stats['tp']} | FP: {stats['fp']} | FN: {stats['fn']} | "
            f"Precision: {stats['precision']:.3f} | Recall: {stats['recall']:.3f} | F1: {stats['f1']:.3f}"
        )
        if stats["mean_detection_delay"] is not None:
            print(
                f"    Mean Detection Delay: {stats['mean_detection_delay']:.1f} | "
                f"Median Detection Delay: {stats['median_detection_delay']:.1f} | "
                f"Mean Signed Latency: {stats['mean_signed_latency']:.1f}"
            )
            print(
                f"    Mean Abs Error: {stats['mean_abs_error']:.1f} | "
                f"Median Abs Error: {stats['median_abs_error']:.1f}"
            )


def logProvider(log, window):
    """Function to provide the sub-logs of each drift."""
    for drift in window.driftsIdentified:
        W0start = drift.eventsSeen - drift.refWinSize
        W0end = drift.eventsSeen
        W1start = drift.eventsSeen
        W1end = drift.eventsSeen + drift.testWinSize

        W0SubLog = log[W0start:W0end]
        W1SubLog = log[W1start:W1end]

        if not os.path.exists("results"):
            os.makedirs("results")

        variables = W0SubLog[0].__dict__["_dict"].keys()

        df1 = pd.DataFrame([[i.get(j) for j in variables] for i in W0SubLog], columns=variables)
        df2 = pd.DataFrame([[i.get(j) for j in variables] for i in W1SubLog], columns=variables)

        indexOfDrift = str(window.driftsIdentified.index(drift))

        xes_exporter.apply(df1, os.path.abspath("results\\drift" + indexOfDrift + "W0.xes"))
        xes_exporter.apply(df2, os.path.abspath("results\\drift" + indexOfDrift + "W1.xes"))


@click.command()
@click.option(
    "-l",
    "--decay_lambda",
    default=0.25,
    show_default=True,
    is_flag=False,
    flag_value=0.25,
    help="Decaying lambda for the older traces when performing a pruning step.",
)
@click.option(
    "-n",
    "--noise",
    default=1,
    show_default=True,
    is_flag=False,
    flag_value=1,
    help="Noise filter applied to relations and activities which have a frequency below certain threshold.",
)
@click.option(
    "-w",
    "--window_size",
    default=10,
    show_default=True,
    is_flag=False,
    flag_value=10,
    help="Maximum size of the window of detection (maxWindowSize)",
)
@click.option(
    "-t",
    "--tree_size",
    default=1000,
    show_default=True,
    is_flag=False,
    flag_value=1000,
    help="Maximum size of the trees (pruningSteps)",
)
@click.option(
    "-c",
    "--config",
    default=False,
    show_default=True,
    is_flag=False,
    flag_value=False,
    help="(True/False) Configuration for generating sub-logs from the drifts identified",
)
@click.option("-f", "--file", required=True, help="Path to the XES log file.")
@click.option(
    "--trace-changepoints",
    multiple=True,
    type=int,
    help="Optional trace indices where drifts start. Use this when the log has no drift metadata.",
)
@click.option(
    "--case-changepoints",
    multiple=True,
    type=int,
    help="Optional case changepoints for synthetic logs. These are converted to trace start indices with +1.",
)
@click.option(
    "--metric-tolerances",
    default="1000,2000,3000,5000",
    show_default=True,
    help="Comma-separated allowed late-event lags used when reporting F1 and latency.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to save the structured benchmark result as JSON.",
)
def main(
    config,
    file,
    window_size,
    tree_size,
    decay_lambda,
    noise,
    trace_changepoints,
    case_changepoints,
    metric_tolerances,
    output_json,
):
    """Run the Prefix-Tree Concept Drift Detection algorithm with metric tracking."""
    if trace_changepoints and case_changepoints:
        raise click.BadParameter("Use either --trace-changepoints or --case-changepoints, not both.")

    resolved_trace_changepoints = (
        [int(idx) + 1 for idx in case_changepoints]
        if case_changepoints
        else (list(trace_changepoints) if trace_changepoints else None)
    )

    result = run_prefix_tree_detector(
        file_path=file,
        window_size=window_size,
        tree_size=tree_size,
        decay_lambda=decay_lambda,
        noise=noise,
        configuration=config,
        drift_trace_indices=resolved_trace_changepoints,
        metric_tolerances=parse_int_csv(metric_tolerances),
    )
    print_evaluation_summary(result)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to {output_json}")


if __name__ == "__main__":
    main()
