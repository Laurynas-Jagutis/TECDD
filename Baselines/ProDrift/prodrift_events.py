"""
Event stream ProDrift, ported from the original Apromore Java implementation
(prodrift-osgi 2.4, ControlFlowDriftDetector_EventStream, fixed window mode with the default
"Alpha+" relations), i.e. the method of Ostovar et al. (2016), "Detecting Drift from Event
Streams of Unpredictable Business Processes".

What is ported, following the Java default path:
  - The event stream: every 'complete' event of the log, stably sorted by timestamp
    (LogStreamer.logStreamer).
  - Two adjacent fixed windows of events. The detection window holds the last `window_size`
    events, the reference window the `window_size` events before it. Inside a window, events
    are grouped by case into partial traces (buildSubLogFromEventStream).
  - Directly follows and two loop (ABA) counts per window
    (BasicLogRelationsExt.fillDirectSuccessionMatrices).
  - The relation noise filter and alpha+ relations: causal, concurrent, length one loop,
    length two loop (BasicLogRelationsExt.makeBasicRelations / filterDFgraph and
    getAlphaRelationFrequency).
  - The two row contingency matrix of relation frequencies
    (build_AlphaRelations_FrequencyMatrix), tested with the G-test of independence as in
    Ostovar et al. (2016).
  - The persistence rule: a drift is a run of at least int(window_size * (1 - sensitivity))
    consecutive p-values <= 0.05 (MeanDelayCurve.retreiveDrift_EventStream).

Deliberate differences from the Java tool:
  - Detections are reported at confirmation, i.e. at the event where the run of significant
    tests reaches the persistence length. The Java tool reports the first significant test
    of the run, which is only known later.
  - Detection is online: each p-value uses only events up to the current one.
  - Window counts are maintained incrementally, with the same definitions as the Java
    computation from scratch on each window's partial traces.
  - `step`: test every `step` events instead of every event (for speed). The persistence
    length is then converted from events to tests with ceil(L / step).
"""
import math
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from scipy.stats import chi2

XES_NS = "{http://www.xes-standard.org/}"

LOOP_THRESHOLD = 0.3     # BasicLogRelationsExt.loopThresh: min ratio of ABA to BA count for a two loop
PVALUE_THRESHOLD = 0.05

CAUSAL = "Causal"
CONCURRENCY = "Concurrency"
LENGTH_ONE_LOOP = "Length_One_Loop"
LENGTH_TWO_LOOP = "Length_Two_Loop"


# --- Log loading -----------------------------------------------------------------------------

@dataclass
class StreamEvent:
    case_id: str
    activity: str
    full_position: int  # 1-based position in the full time-ordered log (every lifecycle transition)


def _parse_timestamp(value):
    ts = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def load_event_stream(xes_path):
    """Read an XES log. Returns (stream, case_count, full_case_indices).

    stream: the 'complete' events (and events without a lifecycle attribute), as in the Java
    LogStreamer, stably sorted by timestamp. Detection only ever sees this stream.
    full_case_indices[i]: the 0-based case of event i + 1 of the full log (every event with a
    timestamp, stably sorted by timestamp), the axis all methods are scored on.
    """
    all_events = []  # (timestamp, case_index, case_id, activity, kept_by_prodrift)
    case_count = 0
    with open(xes_path, "rb") as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if element.tag != f"{XES_NS}trace":
                continue
            case_id = None
            for attr in element:
                if attr.tag == f"{XES_NS}string" and attr.get("key") == "concept:name":
                    case_id = attr.get("value")
                    break
            case_index = case_count
            case_count += 1
            for event in element.findall(f"{XES_NS}event"):
                activity = timestamp = lifecycle = None
                for attr in event:
                    key = attr.get("key")
                    if key == "concept:name":
                        activity = attr.get("value")
                    elif key == "time:timestamp":
                        timestamp = attr.get("value")
                    elif key == "lifecycle:transition":
                        lifecycle = attr.get("value")
                if timestamp is None:
                    continue
                keep = activity is not None and (lifecycle is None or lifecycle.strip().lower() == "complete")
                all_events.append((_parse_timestamp(timestamp), case_index,
                                   case_id if case_id is not None else str(case_index), activity, keep))
            element.clear()

    all_events.sort(key=lambda item: item[0])  # stable: equal timestamps keep file order
    stream, full_case_indices = [], []
    for position, (_, case_index, case_id, activity, keep) in enumerate(all_events, start=1):
        full_case_indices.append(case_index)
        if keep:
            stream.append(StreamEvent(case_id=case_id, activity=activity, full_position=position))
    return stream, case_count, full_case_indices


# --- One window: partial traces and directly follows / two loop counts -----------------------

@dataclass
class _Window:
    cases: dict = field(default_factory=dict)      # case id -> deque of activity ids
    df: Counter = field(default_factory=Counter)   # (a, b) -> directly follows count
    tl: Counter = field(default_factory=Counter)   # (a, b) -> count of a, b, a patterns

    def add_back(self, case, act):
        trace = self.cases.get(case)
        if trace is None:
            trace = deque()
            self.cases[case] = trace
        if trace:
            last = trace[-1]
            self.df[(last, act)] += 1
            if len(trace) >= 2 and trace[-2] == act and last != act:
                self.tl[(act, last)] += 1
        trace.append(act)

    def remove_front(self, case):
        # The event leaving a window is always the oldest event of its case in that window.
        trace = self.cases[case]
        if len(trace) >= 2:
            pair = (trace[0], trace[1])
            self.df[pair] -= 1
            if self.df[pair] == 0:
                del self.df[pair]
            if len(trace) >= 3 and trace[0] != trace[1] and trace[2] == trace[0]:
                self.tl[pair] -= 1
                if self.tl[pair] == 0:
                    del self.tl[pair]
        trace.popleft()
        if not trace:
            del self.cases[case]


# --- Alpha+ relations of one window (BasicLogRelationsExt + getAlphaRelationFrequency) ------

def _window_relations(win, other, noise_threshold, names):
    """{(a, b): (relation type, frequency)} for one window. `other` is the adjacent window. As in
    filterDFgraph, a relation is noise only if it is rare in this window AND absent or rare in
    the other window."""
    df, other_df = win.df, other.df
    max_card = max(df.values()) if df else 0
    max_card_other = max(other_df.values()) if other_df else 0

    def df_is_noise(pair):
        if df[pair] >= max_card * noise_threshold:
            return False
        other_card = other_df.get(pair)
        return other_card is None or other_card < max_card_other * noise_threshold

    df_noise = {pair: df_is_noise(pair) for pair in df}

    tl_ok = set()  # two loop pairs that are not noise
    for pair, loop_freq in win.tl.items():
        opposed = (pair[1], pair[0])
        if df_noise.get(pair, True) or df_noise.get(opposed, True):
            continue
        if loop_freq < df[opposed] * LOOP_THRESHOLD:
            other_loop = other.tl.get(pair)
            if other_loop is None or other_loop < other_df.get(opposed, 0) * LOOP_THRESHOLD:
                continue
        tl_ok.add(pair)

    causal, parallel, self_loop = set(), set(), {}
    for pair, noisy in df_noise.items():
        if noisy:
            continue
        a, b = pair
        opposed = (b, a)
        if not df_noise.get(opposed, True):
            if pair in tl_ok or opposed in tl_ok:
                causal.add(pair)
                causal.add(opposed)
            else:
                if a == b:
                    self_loop[a] = df[pair]
                parallel.add(pair)
                parallel.add(opposed)
        else:
            causal.add(pair)

    relations, seen = {}, set()
    for a, b in df:
        key = (a, b) if a <= b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        a, b = key
        if a == b:
            if a in self_loop:
                relations[(a, a)] = (LENGTH_ONE_LOOP, self_loop[a])
            continue
        if (a, b) in tl_ok:
            relations[(a, b)] = (LENGTH_TWO_LOOP, win.tl[(a, b)])
            if (b, a) in tl_ok:
                relations[(b, a)] = (LENGTH_TWO_LOOP, win.tl[(b, a)])
            elif (b, a) in causal:
                relations[(b, a)] = (CAUSAL, df[(b, a)])
        elif (b, a) in tl_ok:
            relations[(b, a)] = (LENGTH_TWO_LOOP, win.tl[(b, a)])
            if (a, b) in causal:
                relations[(a, b)] = (CAUSAL, df[(a, b)])
        elif (a, b) in parallel:
            concurrency_key = (a, b) if names[a] <= names[b] else (b, a)
            relations[concurrency_key] = (CONCURRENCY, min(df[(a, b)], df[(b, a)]))
        elif (a, b) in causal:
            relations[(a, b)] = (CAUSAL, df[(a, b)])
        elif (b, a) in causal:
            relations[(b, a)] = (CAUSAL, df[(b, a)])
    return relations


def _gtest_pvalue(det_freqs, ref_freqs):
    """G-test of independence on the 2 x k table, on raw counts: G = 2 * sum(O * ln(O / E)),
    df = k - 1. The Java code (Apache GTest inside a try/catch that sets p = 1) fails when a row
    is all zero or a column is zero in both rows, so those cases return 1.0."""
    k = len(det_freqs)
    if k <= 1:
        return 1.0
    row_det, row_ref = float(sum(det_freqs)), float(sum(ref_freqs))
    total = row_det + row_ref
    if row_det <= 0 or row_ref <= 0 or any(d == 0 and r == 0 for d, r in zip(det_freqs, ref_freqs)):
        return 1.0
    g = 0.0
    for d, r in zip(det_freqs, ref_freqs):
        col = d + r
        if d > 0:
            g += d * math.log(d * total / (row_det * col))
        if r > 0:
            g += r * math.log(r * total / (row_ref * col))
    return float(chi2.sf(2.0 * g, k - 1))


def _window_pvalue(det, ref, noise_threshold, names):
    det_rel = _window_relations(det, ref, noise_threshold, names)
    ref_rel = _window_relations(ref, det, noise_threshold, names)
    det_freqs, ref_freqs = [], []
    for pair in det_rel.keys() | ref_rel.keys():
        d, r = det_rel.get(pair), ref_rel.get(pair)
        if d is not None and r is not None and d[0] == r[0]:
            det_freqs.append(d[1])
            ref_freqs.append(r[1])
            continue
        if d is not None:
            det_freqs.append(d[1])
            ref_freqs.append(0)
        if r is not None:
            det_freqs.append(0)
            ref_freqs.append(r[1])
    return _gtest_pvalue(det_freqs, ref_freqs)


# --- P-value series and drift retrieval ------------------------------------------------------

def compute_pvalues(stream, window_size, noise_filter_percentage, step):
    """Slide the two windows over the stream and return [(position, p-value), ...]. position is
    the 0-based stream position of the newest event in the detection window. The first test is
    at position 2 * window_size - 1."""
    n = len(stream)
    if n < 2 * window_size:
        return []

    activity_ids = {}
    acts = [activity_ids.setdefault(ev.activity, len(activity_ids)) for ev in stream]
    names = [None] * len(activity_ids)
    for name, idx in activity_ids.items():
        names[idx] = name
    cases = [ev.case_id for ev in stream]
    noise_threshold = float(noise_filter_percentage) / 100.0

    det, ref = _Window(), _Window()
    for i in range(window_size):
        ref.add_back(cases[i], acts[i])
    for i in range(window_size, 2 * window_size):
        det.add_back(cases[i], acts[i])

    results = []
    position = 2 * window_size - 1
    while True:
        results.append((position, _window_pvalue(det, ref, noise_threshold, names)))
        next_position = position + step
        if next_position >= n:
            break
        for e in range(position + 1, next_position + 1):
            # e enters the detection window, e - w moves to the reference, e - 2w leaves it
            det.add_back(cases[e], acts[e])
            moved = e - window_size
            det.remove_front(cases[moved])
            ref.add_back(cases[moved], acts[moved])
            ref.remove_front(cases[e - 2 * window_size])
        position = next_position
    return results


def persistence_events(window_size, sensitivity):
    """Run length (in events) required for a drift, as in retreiveDrift_EventStream, with
    oscillation = 1 - sensitivity (0.01 -> 1 and 1 -> 0 as special cases)."""
    if math.isclose(sensitivity, 0.01, abs_tol=1e-6):
        oscillation = 1.0
    elif math.isclose(sensitivity, 1.0, abs_tol=1e-6):
        return 1
    else:
        oscillation = 1.0 - float(sensitivity)
    return max(1, int(window_size * oscillation))


def retrieve_drifts(pvalues, window_size, sensitivity, step):
    """Online persistence rule: a drift is confirmed at the test where a run of consecutive
    p-values <= 0.05 reaches the required length. One detection per run. Returns the 0-based
    stream positions of the confirming tests."""
    required_tests = max(1, math.ceil(persistence_events(window_size, sensitivity) / step))
    detections = []
    run_length = 0
    confirmed = False
    for position, p in pvalues:
        if p <= PVALUE_THRESHOLD:
            run_length += 1
            if not confirmed and run_length >= required_tests:
                detections.append(position)
                confirmed = True
        else:
            run_length = 0
            confirmed = False
    return detections
