"""Per case features used to cluster the cases of a base log into blocks."""
from collections import Counter

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from xes_io import nanoseconds

TOP_ACTIVITIES, TOP_DFG_PAIRS, TOP_RESOURCES = 40, 80, 40
MAX_INDEXED_EVENTS = 40              # position features for the first 40 events of a case


def concurrency_features(cases):
    """Per case: open cases at its start, and their time weighted mean and maximum over the case,
    not counting the case itself."""
    starts = np.array([nanoseconds(case.start) for case in cases], dtype=np.int64)
    ends = np.array([nanoseconds(case.end) for case in cases], dtype=np.int64)
    sorted_starts, sorted_ends = np.sort(starts), np.sort(ends)
    change_points = np.unique(np.concatenate([starts, ends]))

    def open_at(ts):
        return np.searchsorted(sorted_starts, ts, side="right") - np.searchsorted(sorted_ends, ts, side="right")

    rows = []
    for start, end in zip(starts, ends):
        if end == start:
            rows.append((float(open_at(start)),) * 3)
            continue
        inner = change_points[np.searchsorted(change_points, start, side="right"):np.searchsorted(change_points, end)]
        bounds = np.concatenate(([start], inner, [end]))
        counts = (open_at(bounds[:-1]) - 1).tolist()
        weighted = 0.0
        for count, width in zip(counts, np.diff(bounds).tolist()):
            weighted += count * width
        rows.append((float(counts[0]), weighted / float(end - start), float(max(counts))))
    return rows


def feature_matrix(cases):
    """Standardized per case features: length, distinct activities and resources, duration, time
    gaps, concurrency, start time, relative frequencies of the most frequent activities, directly
    follows pairs and resources, and which activity and resource occur at each of the first 40
    positions."""
    counts, totals = [], (Counter(), Counter(), Counter())
    for case in cases:
        activities = [activity for _, activity, _, _ in case.events]
        resources = [resource for _, _, resource, _ in case.events]
        case_counts = (Counter(activities), Counter(zip(activities, activities[1:])), Counter(resources))
        for total, part in zip(totals, case_counts):
            total.update(part)
        counts.append((activities, resources, case_counts))
    keep_act, keep_dfg, keep_res = (
        {name for name, _ in total.most_common(n)} for total, n in zip(totals, (TOP_ACTIVITIES, TOP_DFG_PAIRS, TOP_RESOURCES))
    )

    first_start = min(case.start for case in cases)
    rows = []
    for case, (activities, resources, (act_counts, dfg_counts, res_counts)), opened in zip(
            cases, counts, concurrency_features(cases)):
        gaps = np.diff(np.array([nanoseconds(ts) for ts, *_ in case.events], dtype=np.int64)) / 3.6e12
        n = len(activities)
        row = {
            "trace_length": float(n),
            "unique_activities": float(len(set(activities))),
            "unique_resources": float(len(set(resources))),
            "log_duration_hours": float(np.log1p((case.end - case.start).total_seconds() / 3600.0)),
            "log_mean_gap_hours": float(np.log1p(gaps.mean() if len(gaps) else 0.0)),
            "log_std_gap_hours": float(np.log1p(gaps.std() if len(gaps) else 0.0)),
            "log_open_cases_at_trace_start": float(np.log1p(opened[0])),
            "log_mean_open_cases_during_trace": float(np.log1p(opened[1])),
            "log_max_open_cases_during_trace": float(np.log1p(opened[2])),
            "start_offset_days": (case.start - first_start).total_seconds() / 86400.0,
            "start_weekday": float(case.start.weekday()),
        }
        row.update({f"act::{a}": c / n for a, c in act_counts.items() if a in keep_act})
        row.update({f"dfg::{a}->{b}": c / max(n - 1, 1) for (a, b), c in dfg_counts.items() if (a, b) in keep_dfg})
        row.update({f"res::{r}": c / n for r, c in res_counts.items() if r in keep_res})
        for position, (activity, resource) in enumerate(zip(activities[:MAX_INDEXED_EVENTS], resources), start=1):
            if activity in keep_act:
                row[f"idx::{position:03d}::act::{activity}"] = 1.0
            if resource in keep_res:
                row[f"idx::{position:03d}::res::{resource}"] = 1.0
        rows.append(row)
    return StandardScaler().fit_transform(DictVectorizer(sparse=False).fit_transform(rows))
