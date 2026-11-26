"""Build one semi synthetic log of the paper (validation and test split) from a base log.

1. The cases of the base log are clustered with k-means (k = 25) on PCA reduced trace features,
   clusters that lie very close together are merged, and clusters below 50 cases or 500 events
   are dropped. Every remaining cluster is a block of cases.
2. The blocks are replayed one after another. After each block, the next one is the unused
   block farthest from it, at median distance from it or nearest to it (--order).
3. The drift type (--drift-type) only decides when the cases start:
   simultaneous  all cases of a block start at the block boundary,
   staggered     the cases of a block keep the start gaps they had in the base log,
   gradual       the last cases of one block and the first of the next are interleaved over a
                 transition, with the share of new cases rising linearly from 0 to 1.
4. The log is split by blocks into a validation log (first half of the drifts) and a test log
   (the rest). The drift at the split point is dropped.

Output: <output-dir>/{validation,test}/event_log.xes and drift_ground_truth.csv, plus
<output-dir>/metadata.json with the block order and the split.
"""
import argparse
import csv
import json
import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from trace_features import feature_matrix
from xes_io import by_start, copy_case, load_base_log, nanoseconds, write_xes

SEED = 42
NUM_CLUSTERS = 25
PCA_COMPONENTS = 12
KMEANS_N_INIT = 50
MERGE_THRESHOLD_FRACTION = 0.005     # merge clusters closer than 0.5% of the largest cluster distance
MIN_BLOCK_CASES = 50
MIN_BLOCK_EVENTS = 500
MAX_DRIFTS = 24
TRANSITION_TARGET_EVENTS = 2000      # gradual: events per transition, both blocks together
TRANSITION_MAX_FRACTION = 0.4        # gradual: at most 40% of the smaller block
SYNTHETIC_LOG_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def distance_matrix(clusters, features):
    """Euclidean distances between the mean features of the clusters."""
    profiles = np.vstack([features[[case.row for case in cluster]].mean(axis=0) for cluster in clusters])
    diff = profiles[:, None, :] - profiles[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(distances, 0.0)
    return distances


def build_blocks(cases, features):
    """k-means on the PCA reduced features, then clusters closer than MERGE_THRESHOLD_FRACTION of
    the largest distance are merged (transitively). Merged clusters are ordered by their earliest
    case. Ties (cases starting at the same time) go to the group whose k-means cluster comes first
    when the k-means clusters are numbered by median case start, median duration and size."""
    embedding = PCA(n_components=PCA_COMPONENTS, random_state=SEED).fit_transform(features)
    labels = KMeans(n_clusters=NUM_CLUSTERS, random_state=SEED, n_init=KMEANS_N_INIT).fit_predict(embedding)
    clusters = [by_start(c for c, l in zip(cases, labels) if l == label) for label in sorted(set(labels))]
    clusters.sort(key=lambda cluster: (
        float(np.median([nanoseconds(case.start) for case in cluster])),
        float(np.median([(case.end - case.start).total_seconds() / 3600.0 for case in cluster])),
        -len(cluster),
    ))
    names = [f"V1{number}" for number in range(1, len(clusters) + 1)]   # the tie breaker compares these as text

    distances = distance_matrix(clusters, features)
    pairs = list(zip(*np.triu_indices(len(clusters), k=1)))
    threshold = max(distances[i, j] for i, j in pairs) * MERGE_THRESHOLD_FRACTION
    group = list(range(len(clusters)))
    for i, j in pairs:
        if distances[i, j] < threshold and group[i] != group[j]:
            old = group[j]
            group = [group[i] if g == old else g for g in group]
    merged = [
        (by_start(case for k, cluster in enumerate(clusters) if group[k] == g for case in cluster),
         min(names[k] for k in range(len(clusters)) if group[k] == g))
        for g in sorted(set(group))
    ]
    return [cluster for cluster, _ in sorted(merged, key=lambda item: (item[0][0].start, item[1]))]


def block_order(eligible, distances, order):
    """Start with the first eligible block and repeatedly append the unused block farthest from,
    at median distance from or nearest to the last one."""
    sequence, remaining = [eligible[0]], eligible[1:]
    while remaining and len(sequence) <= MAX_DRIFTS:
        ranked = sorted(remaining, key=lambda k: distances[sequence[-1], k])
        pick = {"farthest": ranked[-1], "median": ranked[len(ranked) // 2], "nearest": ranked[0]}[order]
        sequence.append(pick)
        remaining.remove(pick)
    return sequence


# --- Synthetic log ---------------------------------------------------------------------------

def build_sudden_log(blocks, gap, simultaneous):
    """Blocks one after another, each starting one gap after the previous block ended. Within a
    block, cases start at the boundary (simultaneous) or keep their original offsets (staggered)."""
    cases, anchor = [], SYNTHETIC_LOG_START
    for block_index, block in enumerate(blocks, start=1):
        copies = [
            copy_case(source, f"C{len(cases) + i:06d}", block_index,
                      anchor + (timedelta(0) if simultaneous else source.start - block[0].start) - source.start)
            for i, source in enumerate(block, start=1)
        ]
        cases.extend(copies)
        anchor = max(case.end for case in copies) + gap
    return cases


def transition_half_size(old_cases, new_cases):
    """Cases each block gives to the transition: TRANSITION_TARGET_EVENTS events in total, capped
    at TRANSITION_MAX_FRACTION of the smaller block."""
    cap = int(TRANSITION_MAX_FRACTION * min(len(old_cases), len(new_cases)) / 2)
    mean_old = sum(len(case.events) for case in old_cases) / len(old_cases)
    mean_new = sum(len(case.events) for case in new_cases) / len(new_cases)
    return max(0, min(round(TRANSITION_TARGET_EVENTS / (mean_old + mean_new)), cap))


def mix_transition(old_cases, new_cases, rng):
    """Interleave the old block's tail with the new block's head so the share of new cases rises
    linearly: old cases get positions drawn with density 2(1-x), new cases with density 2x."""
    old_positions = sorted(1.0 - math.sqrt(rng.random()) for _ in old_cases)
    new_positions = sorted(math.sqrt(rng.random()) for _ in new_cases)
    tagged = [(pos, 0, i, case) for i, (pos, case) in enumerate(zip(old_positions, old_cases))]
    tagged += [(pos, 1, i, case) for i, (pos, case) in enumerate(zip(new_positions, new_cases))]
    return [case for *_, case in sorted(tagged, key=lambda item: item[:3])]


def build_gradual_log(blocks, gap):
    """Cases queued as: pure part of block 1, transition 1->2, pure part of block 2, ... Every case
    starts its original start gap after the previous case. The first case of a block uses gap."""
    rng = random.Random(SEED)
    halves = [transition_half_size(a, b) for a, b in zip(blocks, blocks[1:])]
    queued = [
        [(case, block_index, g) for case, g in zip(block, [gap] + [b.start - a.start for a, b in zip(block, block[1:])])]
        for block_index, block in enumerate(blocks, start=1)
    ]
    queue = []
    for k, block in enumerate(queued):
        head = halves[k - 1] if k > 0 else 0
        tail = halves[k] if k < len(halves) else 0
        queue.extend(block[head:len(block) - tail])
        if k < len(halves):
            queue.extend(mix_transition(block[len(block) - tail:], queued[k + 1][:tail], rng))

    cases, start = [], SYNTHETIC_LOG_START
    for number, (source, block_index, g) in enumerate(queue, start=1):
        start = start + g if number > 1 else start
        cases.append(copy_case(source, f"C{number:06d}", block_index, start - source.start))
    return cases


def ground_truth(cases, gradual):
    """One row per block boundary of the split log, as 0-based case and event positions in time
    order. A drift starts at the first case of the new block. For gradual drifts it ends at the
    last case of the old block (transition end)."""
    ordered = by_start(cases)
    case_position = {case.case_id: i for i, case in enumerate(ordered)}
    first_event = {}
    for position, (_, case_id, _) in enumerate(sorted(
            (ts, case.case_id, i) for case in ordered for i, (ts, *_) in enumerate(case.events))):
        first_event.setdefault(case_id, position)

    by_block = {}
    for case in ordered:
        by_block.setdefault(case.block_index, []).append(case)
    blocks = sorted(by_block)
    rows = []
    for drift_id, (old, new) in enumerate(zip(blocks, blocks[1:]), start=1):
        first_new = by_block[new][0]
        row = {
            "drift_id": drift_id,
            "from_variant_id": by_block[old][0].variant_id,
            "to_variant_id": first_new.variant_id,
            "drift_timestamp": first_new.start.isoformat(),
            "drift_case_index_0based": case_position[first_new.case_id],
            "drift_event_index_0based": first_event[first_new.case_id],
        }
        if gradual:
            last_old = by_block[old][-1]
            end = last_old if case_position[last_old.case_id] > case_position[first_new.case_id] else first_new
            row["transition_end_timestamp"] = end.start.isoformat()
            row["transition_end_case_index_0based"] = case_position[end.case_id]
            row["transition_end_event_index_0based"] = first_event[end.case_id]
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Build one semi synthetic log (validation and test split).")
    parser.add_argument("--input-xes", type=Path, required=True, help="Base log (BPIC 2012, or the merged BPIC 2015).")
    parser.add_argument("--order", choices=("farthest", "median", "nearest"), required=True)
    parser.add_argument("--drift-type", choices=("simultaneous", "staggered", "gradual"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"Loading {args.input_xes}...", flush=True)
    root_attrib, header, cases = load_base_log(args.input_xes)
    print(f"{len(cases):,} cases, {sum(len(case.events) for case in cases):,} events", flush=True)

    # Gap between blocks: the median positive gap between consecutive case starts.
    starts = sorted(case.start for case in cases)
    gap_hours = float(np.median([g for a, b in zip(starts, starts[1:]) if (g := (b - a).total_seconds() / 3600.0) > 0]))
    gap = timedelta(hours=gap_hours)

    features = feature_matrix(cases)
    clusters = build_blocks(cases, features)
    for number, cluster in enumerate(clusters, start=1):
        for case in cluster:
            case.variant_id = f"V1{number}"
    eligible = [k for k, cluster in enumerate(clusters)
                if len(cluster) >= MIN_BLOCK_CASES and sum(len(case.events) for case in cluster) >= MIN_BLOCK_EVENTS]
    print(f"{len(clusters)} clusters after merging, {len(eligible)} with at least "
          f"{MIN_BLOCK_CASES} cases and {MIN_BLOCK_EVENTS} events", flush=True)
    sequence = [clusters[k] for k in block_order(eligible, distance_matrix(clusters, features), args.order)]
    sequence_ids = [block[0].variant_id for block in sequence]
    print("Block order: " + " -> ".join(sequence_ids), flush=True)

    if args.drift_type == "gradual":
        synthetic = build_gradual_log(sequence, gap)
    else:
        synthetic = build_sudden_log(sequence, gap, simultaneous=args.drift_type == "simultaneous")

    # Validation gets the first half of the drifts, test the rest. The drift between is dropped.
    validation_drifts = round((len(sequence) - 1) / 2)
    block_numbers = list(range(1, len(sequence) + 1))
    splits = {"validation": block_numbers[:validation_drifts + 1], "test": block_numbers[validation_drifts + 1:]}
    for split_name, block_indices in splits.items():
        split_cases = [case for case in synthetic if case.block_index in block_indices]
        split_dir = args.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        write_xes(split_dir / "event_log.xes", split_cases, root_attrib, header,
                  log_name=f"{args.input_xes.stem}_{args.order}_{args.drift_type}_{split_name}")
        rows = ground_truth(split_cases, gradual=args.drift_type == "gradual")
        with (split_dir / "drift_ground_truth.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split_name}: {len(split_cases):,} cases, {sum(len(case.events) for case in split_cases):,} events, "
              f"{len(rows)} drifts -> {split_dir}", flush=True)

    metadata = {
        "input_xes": str(args.input_xes),
        "order": args.order,
        "drift_type": args.drift_type,
        "block_sequence": sequence_ids,
        "split_block_indices": splits,
        "inter_block_gap_hours": gap.total_seconds() / 3600.0,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    main()
