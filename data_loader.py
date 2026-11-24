import xml.etree.ElementTree as ET

import numpy as np

from case_buffer import CaseBuffer
from event import Event

XES_NS = {"xes": "http://www.xes-standard.org/"}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def load_xes_traces(xes_path):
    """All traces of an XES log as (case id, events), plus the set of activities.
    Every event also gets its 1-based position in the time-ordered stream."""
    root = ET.parse(xes_path).getroot()
    traces = []
    vocab_set = set()
    for trace in root.findall("xes:trace", XES_NS):
        case_id = trace.find('xes:string[@key="concept:name"]', XES_NS).attrib["value"]
        events = []
        for ev in trace.findall("xes:event", XES_NS):
            act_el = ev.find('xes:string[@key="concept:name"]', XES_NS)
            if act_el is None:
                continue
            ts_el = ev.find('xes:date[@key="time:timestamp"]', XES_NS)
            ts = ts_el.attrib["value"] if ts_el is not None else "1970-01-01T00:00:00"
            events.append(Event(case_id, act_el.attrib["value"], ts))
            vocab_set.add(act_el.attrib["value"])
        if events:
            traces.append((case_id, events))

    all_events = [event for _, events in traces for event in events]
    for global_event_index, event in enumerate(sorted(all_events, key=lambda ev: ev.timestamp), start=1):
        event.global_event_index = global_event_index
    return traces, vocab_set


def build_activity_vocab(vocab_set):
    """<PAD> = 0, <UNK> = 1, then the activities in sorted order."""
    idx_to_act = [PAD_TOKEN, UNK_TOKEN] + sorted(vocab_set - {PAD_TOKEN, UNK_TOKEN})
    act_to_idx = {act: i for i, act in enumerate(idx_to_act)}
    return act_to_idx, idx_to_act


def traces_to_supervised_dataset(trace_list, prefix_len, act_to_idx):
    buffer = CaseBuffer(prefix_len)
    pad_id = act_to_idx[PAD_TOKEN]
    unk_id = act_to_idx.get(UNK_TOKEN, pad_id)
    X_acts, X_times, y_acts, y_times = [], [], [], []
    
    for case_id, events in trace_list:
        buffer.remove(case_id)
        for j in range(len(events) - 1):
            buffer.add(events[j])
            prefix_acts, prefix_times = buffer.get_prefix_for_a_case(case_id)
            pad_len = prefix_len - len(prefix_acts)
            X_acts.append([pad_id] * pad_len + [act_to_idx.get(a, unk_id) for a in prefix_acts])
            X_times.append([0.0] * pad_len + prefix_times)
            y_acts.append(act_to_idx.get(events[j + 1].activity, unk_id))
            y_times.append((events[j + 1].timestamp - events[j].timestamp).total_seconds())
            
            
    return (np.array(X_acts), np.array(X_times, dtype=np.float32), np.array(y_acts), np.array(y_times, dtype=np.float32))


def traces_to_replay_stream(trace_list):
    events = [event for _, events in trace_list for event in events]
    return sorted(events, key=lambda event: event.global_event_index)
