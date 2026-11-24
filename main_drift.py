"""TECDD online monitoring (Stage 3): cold start, persistence rule, reference search and scoring."""
import csv
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from case_buffer import CaseBuffer
from data_loader import load_xes_traces, traces_to_replay_stream
from drift_detector import DriftDetector
from models.sutran import load_checkpoint
from prefix_embedder import PrefixEmbedder
from time_preprocessing import TIME_PREPROCESSING_VERSION, build_padded_time_row, sanitize_time_std
from window_processor import StrideWindowProcessor

XES_NS = {"xes": "http://www.xes-standard.org/"}
PREFIX_LEN = 10                       # prefix length K

# Fixed settings of the final method. Only W, s, m and B are tuned.
COOLDOWN_WINDOWS = 100                # windows without further reports after a drift
WARMUP_EVENTS = 250                   # events before activation replayed to fill the case buffers
STABILITY_THRESHOLD_FRACTION = 0.75   # agreement check: max(theta', 0.75 * theta)
INTERRUPT_MULTIPLIER = 1.5            # a window "clearly leaves" the bank above 1.5x its threshold
INTERRUPT_CONSECUTIVE = 2
SEARCH_DELAY_WINDOWS = 3              # sampled windows skipped after a drift
SEARCH_STABILITY_WINDOWS = 4          # most recent sampled windows that must agree after a drift
COLD_START_SAMPLE_INTERVAL = 3        # every third window at stream start
COLD_START_BANK_SIZE = 50             # B at stream start
COLD_START_STABILITY_WINDOWS = 3
COLD_START_DELAY_WINDOWS = 2
SCORE_TOLERANCE = 500                 # events after g_end that still count as a true positive


# --- Stream and ground truth -------------------------------------------------------------

def load_ground_truth(ground_truth_csv):
    """Drift start cases and, for gradual drifts, the last old case of each transition (0-based)."""
    with Path(ground_truth_csv).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    drift_trace_indices = [int(row["drift_case_index_0based"]) for row in rows]
    transition_end_trace_indices = (
        [int(row["transition_end_case_index_0based"]) for row in rows]
        if rows and "transition_end_case_index_0based" in rows[0]
        else None
    )
    return drift_trace_indices, transition_end_trace_indices


def _ordered_xes_events(xes_path):
    """(timestamp, file position, trace index) of every event, in stream order."""
    ordered_events = []
    for trace_idx, trace in enumerate(ET.parse(xes_path).getroot().findall("xes:trace", XES_NS)):
        for event in trace.findall("xes:event", XES_NS):
            ts_el = event.find('xes:date[@key="time:timestamp"]', XES_NS)
            ts_value = ts_el.attrib["value"] if ts_el is not None else "1970-01-01T00:00:00"
            timestamp = datetime.fromisoformat(ts_value.replace("Z", "+00:00"))
            ordered_events.append((timestamp, len(ordered_events), trace_idx))
    ordered_events.sort(key=lambda item: (item[0], item[1]))
    return ordered_events


def compute_true_global_drift_indices(xes_path, drift_trace_indices):
    """1-based stream position of the first event of each drift's block of cases."""
    ordered_events = _ordered_xes_events(xes_path)
    trace_count = len(ET.parse(xes_path).getroot().findall("xes:trace", XES_NS))
    true_global_drifts = []
    for idx, start_trace_idx in enumerate(drift_trace_indices):
        end_trace_idx = drift_trace_indices[idx + 1] if idx + 1 < len(drift_trace_indices) else trace_count
        for global_event_index, (_, _, trace_idx) in enumerate(ordered_events, start=1):
            if start_trace_idx <= trace_idx < end_trace_idx:
                true_global_drifts.append(global_event_index)
                break
    return true_global_drifts


def compute_true_global_transition_ends(xes_path, transition_end_trace_indices):
    """1-based stream position of the first event of each transition's last old case."""
    first_global_index = {}
    for global_event_index, (_, _, trace_idx) in enumerate(_ordered_xes_events(xes_path), start=1):
        first_global_index.setdefault(trace_idx, global_event_index)
    return [first_global_index[idx] for idx in transition_end_trace_indices]


# --- Embeddings ----------------------------------------------------------------------------

def build_stable_window_embeddings(stream_events, processor, embedder, warmup_windows):
    """Embeddings of every window over the stream, skipping the first warmup_windows windows."""
    stable_window_embeddings = []
    for stream_event in stream_events:
        w_data = processor.observe_event(stream_event)
        if not w_data or w_data["window_id"] < warmup_windows:
            continue
        stable_window_embeddings.append(embedder.get_embeddings(w_data["X_acts"], w_data["X_times"]))
    return stable_window_embeddings


def build_prefix_embeddings_from_stream(stream_events, prefix_len, act_to_idx, time_mean, time_std, embedder):
    """The embedding of every event of the stream, given its running prefix."""
    if not stream_events:
        return np.array([])

    buffer = CaseBuffer(prefix_len)
    pad_id = act_to_idx["<PAD>"]
    unk_id = act_to_idx.get("<UNK>", pad_id)
    act_rows = []
    time_rows = []
    for stream_event in stream_events:
        buffer.add(stream_event)
        prefix_acts, prefix_times = buffer.get_prefix_for_a_case(stream_event.case_id)
        pad_len = prefix_len - len(prefix_acts)
        act_rows.append([pad_id] * pad_len + [act_to_idx.get(act, unk_id) for act in prefix_acts])
        time_rows.append(build_padded_time_row(prefix_times, prefix_len=prefix_len, time_mean=time_mean, time_std=time_std))

    return embedder.get_embeddings(
        torch.tensor(act_rows, dtype=torch.long),
        torch.tensor(time_rows, dtype=torch.float32),
    )


# --- Reference search ------------------------------------------------------------------------

def make_detection(w_data, warmup_count):
    """A drift report at the last event of the current window."""
    return {
        "window": w_data["window_id"],
        "confirmation_event_idx": max(0, int(w_data["event_index"]) - warmup_count),
        "confirmation_global_event_idx": w_data["global_event_index"],
    }


class ReferenceSearch:
    """Builds a new reference sample from the stream (at its start and after every drift).

    Sampled windows first have to agree (no MMD above max(theta', 0.75 theta) to their pooled
    points). They then start a bank, further sampled windows join it only if they are close to
    it, and a full bank of B windows is pooled into the new reference sample. Two consecutive
    sampled windows that clearly leave the bank restart the search and report a further drift.
    """

    def __init__(self, detector, sample_interval, stability_windows, delay_windows, bank_size,
                 interrupt_min_bank_windows, warmup_count):
        self.detector = detector
        self.sample_interval = sample_interval
        self.stability_windows = stability_windows
        self.delay_windows = delay_windows
        self.bank_size = bank_size
        self.interrupt_min_bank_windows = interrupt_min_bank_windows
        self.warmup_count = warmup_count
        self.active = False
        self._reset()

    def _reset(self):
        self.phase = None
        self.bank = []
        self.settle_buffer = []
        self.last_sampled_window = None
        self.delay_sample_count = 0
        self.interrupt_run = 0

    def start(self, w_id, reason):
        self._reset()
        self.active = True
        self.phase = "delay"
        print(f"[Window {w_id}] {reason}")

    def observe(self, w_data, curr_emb, in_cooldown, threshold):
        detector = self.detector
        w_id = w_data["window_id"]

        should_sample = self.last_sampled_window is None or (w_id - self.last_sampled_window) >= self.sample_interval
        if should_sample:
            self.last_sampled_window = w_id

            if self.phase == "delay":
                # Wait for a few sampled windows, then for the most recent ones to agree.
                self.delay_sample_count += 1
                self.settle_buffer.append(curr_emb.copy())
                if len(self.settle_buffer) > self.stability_windows:
                    self.settle_buffer.pop(0)
                if self.delay_sample_count >= self.delay_windows and len(self.settle_buffer) >= self.stability_windows:
                    settle_stats = detector.calibrate_reference_bank(self.settle_buffer)
                    agreement_threshold = max(detector.reference_update_threshold, threshold * STABILITY_THRESHOLD_FRACTION)
                    if float(np.max(settle_stats["distances"])) <= agreement_threshold:
                        self.phase = "collect"
                        self.bank = [window.copy() for window in self.settle_buffer]
                        self.settle_buffer = []
                        self.interrupt_run = 0
                        print(f"[Window {w_id}] Sampled windows agree, starting the bank.")
                return None

            provisional_stats = detector.calibrate_reference_bank(self.bank)
            bank_dist = detector.distance(curr_emb, provisional_stats["reference_sample"])

            if not in_cooldown and len(self.bank) >= self.interrupt_min_bank_windows:
                interrupt_threshold = max(provisional_stats["threshold"] * INTERRUPT_MULTIPLIER,
                                          detector.reference_update_threshold)
                self.interrupt_run = self.interrupt_run + 1 if bank_dist > interrupt_threshold else 0
                if self.interrupt_run >= INTERRUPT_CONSECUTIVE:
                    print(f"[Window {w_id}] CONFIRMED DRIFT DURING REFERENCE SEARCH (Dist: {bank_dist:.4f})")
                    detection = make_detection(w_data, self.warmup_count)
                    self.start(w_id, "Restarting the reference search after a further drift.")
                    return ("interrupt", detection)

            # Join the bank only if close to it.
            admission_threshold = max(provisional_stats["threshold"], detector.reference_update_threshold)
            if not bank_dist > admission_threshold:
                self.bank.append(curr_emb.copy())
                self.interrupt_run = 0

        if len(self.bank) >= self.bank_size:
            stats = detector.calibrate_reference_bank(self.bank)
            self.active = False
            self._reset()
            print(f"[Window {w_id}] Reference search finished, reference updated.")
            return ("finalized", stats["reference_sample"])

        return None


def cold_start_calibrate(stream_events, detector, embedder, processor, interrupt_min_bank_windows):
    """Find the first stable regime of the stream and return (activation event, reference sample).

    An interrupt during cold start only restarts the search, since monitoring has not begun.
    """
    search = ReferenceSearch(
        detector,
        sample_interval=COLD_START_SAMPLE_INTERVAL,
        stability_windows=COLD_START_STABILITY_WINDOWS,
        delay_windows=COLD_START_DELAY_WINDOWS,
        bank_size=COLD_START_BANK_SIZE,
        interrupt_min_bank_windows=interrupt_min_bank_windows,
        warmup_count=0,
    )
    search.start(0, "Cold start: searching for the first stable regime.")
    fallback_windows = []
    fallback_activation_index = None
    for stream_event in stream_events:
        w_data = processor.observe_event(stream_event)
        if not w_data:
            continue
        curr_emb = embedder.get_embeddings(w_data["X_acts"], w_data["X_times"])
        if len(fallback_windows) < COLD_START_BANK_SIZE:
            fallback_windows.append(curr_emb.copy())
            fallback_activation_index = int(w_data["event_index"])
        result = search.observe(w_data, curr_emb, in_cooldown=False, threshold=detector.threshold)
        if result is not None and result[0] == "finalized":
            return int(w_data["event_index"]), result[1]

    # The stream never settled: anchor the reference on its first COLD_START_BANK_SIZE windows.
    if fallback_windows:
        print("WARNING: cold start found no stable regime, using the first windows instead.")
        return fallback_activation_index, detector.calibrate_reference_bank(fallback_windows)["reference_sample"]
    raise RuntimeError("Cold start could not form a reference (stream too short to fill a window).")


# --- Scoring ---------------------------------------------------------------------------------

def score_detections(true_starts, true_ends, detections, unmonitored_fn=0, tolerance=SCORE_TOLERANCE):
    """A detection d is a true positive if start <= d <= end + tolerance. Detections and drifts
    are matched one to one in stream order. Drifts before activation count as false negatives."""
    ends_by_start = {int(s): max(int(s), int(e)) for s, e in zip(true_starts, true_ends)}
    detected = [int(d["confirmation_event_idx"]) for d in detections]
    i = j = tp = fp = fn = 0
    delays = []
    while i < len(true_starts) and j < len(detected):
        true_idx = int(true_starts[i])
        det_idx = detected[j]
        if det_idx >= true_idx and det_idx <= ends_by_start[true_idx] + tolerance:
            tp += 1
            delays.append(max(0, det_idx - true_idx))
            i += 1
            j += 1
        elif det_idx < true_idx:
            fp += 1
            j += 1
        else:
            fn += 1
            i += 1
    fn += len(true_starts) - i + int(unmonitored_fn)
    fp += len(detected) - j

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay_events": float(np.mean(delays)) if delays else None,
    }


# --- Pipeline --------------------------------------------------------------------------------

def run_pipeline(xes_path, checkpoint_path, ground_truth_csv, window_size, stride, persistence, bank_size):
    run_start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xes_path = Path(xes_path)

    model, ckpt = load_checkpoint(str(checkpoint_path), device=device)
    extra = ckpt.get("extra", {})
    if extra.get("time_preprocessing_version") != TIME_PREPROCESSING_VERSION:
        raise ValueError(f"Checkpoint {checkpoint_path} uses an unsupported time preprocessing version.")
    time_mean = float(extra["time_mean"])
    time_std = sanitize_time_std(extra["time_std"])
    detector = DriftDetector.from_state(extra["detector_state"])
    threshold = detector.threshold

    print(f"Loading data from {xes_path}...")
    drift_trace_indices, transition_end_trace_indices = load_ground_truth(ground_truth_csv)
    traces, _ = load_xes_traces(xes_path)
    act_to_idx = extra["act_to_idx"]  # activities unseen in training map to <UNK>
    full_sorted_stream = traces_to_replay_stream(traces)
    true_global_drifts = compute_true_global_drift_indices(xes_path, drift_trace_indices)
    true_full_stream_drifts = [int(global_idx) - 1 for global_idx in true_global_drifts]

    embedder = PrefixEmbedder(model, device=device)
    interrupt_min_bank_windows = max(3, min(5, bank_size // 2))

    def new_processor():
        return StrideWindowProcessor(
            PREFIX_LEN, window_size, act_to_idx, time_mean=time_mean, time_std=time_std, stride=stride,
        )

    # Cold start: find the first stable regime and its reference sample, without labels.
    activation_event_index, reference = cold_start_calibrate(
        full_sorted_stream, detector, embedder, new_processor(), interrupt_min_bank_windows
    )
    stable_replay_events = full_sorted_stream[:activation_event_index]
    monitor_stream_events = full_sorted_stream[activation_event_index:]
    monitor_true_drifts = [d - activation_event_index for d in true_full_stream_drifts if d >= activation_event_index]
    initialization_true_drifts = [d for d in true_full_stream_drifts if d < activation_event_index]
    print(f"Activation event: {activation_event_index}, drifts before it (counted as FN): {initialization_true_drifts}")

    # Fill the case buffers with the last events before activation.
    processor = new_processor()
    warmup_count = min(WARMUP_EVENTS, len(stable_replay_events) - 1) if len(stable_replay_events) > 1 else 0
    for stream_event in (stable_replay_events[-warmup_count:] if warmup_count > 0 else []):
        processor.observe_event(stream_event)

    print(f"\n--- Monitoring (W={window_size}, s={stride}, m={persistence}, B={bank_size}, theta={threshold:.6f}) ---")
    search = ReferenceSearch(
        detector,
        sample_interval=max(1, (window_size + stride - 1) // stride),
        stability_windows=SEARCH_STABILITY_WINDOWS,
        delay_windows=SEARCH_DELAY_WINDOWS,
        bank_size=max(1, bank_size),
        interrupt_min_bank_windows=interrupt_min_bank_windows,
        warmup_count=warmup_count,
    )
    cooldown_counter = 0
    consecutive = 0
    detected_drifts = []

    for stream_event in monitor_stream_events:
        w_data = processor.observe_event(stream_event)
        if not w_data:
            continue
        w_id = w_data["window_id"]
        curr_emb = embedder.get_embeddings(w_data["X_acts"], w_data["X_times"])

        in_cooldown = cooldown_counter > 0
        if in_cooldown:
            cooldown_counter -= 1

        if search.active:
            result = search.observe(w_data, curr_emb, in_cooldown, threshold)
            if result is not None:
                kind, payload = result
                if kind == "interrupt":
                    detected_drifts.append(payload)
                    cooldown_counter = COOLDOWN_WINDOWS
                else:
                    reference = payload
            continue

        if in_cooldown:
            consecutive = 0
            continue

        # Persistence rule: m consecutive windows above theta, one window below resets the count.
        dist = detector.distance(curr_emb, reference)
        consecutive = consecutive + 1 if dist > threshold else 0
        if consecutive >= persistence:
            print(f"[Window {w_id}] CONFIRMED DRIFT (Dist: {dist:.4f})")
            detected_drifts.append(make_detection(w_data, warmup_count))
            cooldown_counter = COOLDOWN_WINDOWS
            search.start(w_id, "Drift reported, searching for a new reference.")
            consecutive = 0

    # Gradual drifts count as detected anywhere from their start to the transition end + tolerance.
    transition_length_by_global = {}
    if transition_end_trace_indices is not None:
        global_ends = compute_true_global_transition_ends(xes_path, transition_end_trace_indices)
        transition_length_by_global = {
            int(start): max(0, int(end) - int(start)) for start, end in zip(true_global_drifts, global_ends)
        }
    # monitor position (0-based) -> global position (1-based): + activation + 1
    monitor_true_drift_ends = [
        int(m) + transition_length_by_global.get(int(m) + int(activation_event_index) + 1, 0)
        for m in monitor_true_drifts
    ]
    metrics = score_detections(
        monitor_true_drifts, monitor_true_drift_ends, detected_drifts, unmonitored_fn=len(initialization_true_drifts)
    )

    runtime_seconds = time.perf_counter() - run_start_time
    monitored_event_count = len(monitor_stream_events)
    runtime_per_event = runtime_seconds / monitored_event_count if monitored_event_count > 0 else None
    print(
        f"\nTP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']} | F1: {metrics['f1']:.3f} | "
        f"Mean Delay: {metrics['mean_delay_events']} | Runtime: {runtime_seconds:.2f} s"
    )

    embedder.remove_hook()
    return {
        "metrics": metrics,
        "detected_drifts": detected_drifts,
        "true_full_stream_drifts": true_full_stream_drifts,
        "true_drift_count": len(true_full_stream_drifts),
        "activation_event_index": activation_event_index,
        "threshold": threshold,
        "runtime_seconds": runtime_seconds,
        "runtime_per_event": runtime_per_event,
        "monitored_event_count": monitored_event_count,
        "training_runtime_seconds": extra.get("training_runtime_seconds"),
    }
