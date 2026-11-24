from collections import deque

import torch

from case_buffer import CaseBuffer
from event import Event
from time_preprocessing import build_padded_time_row, sanitize_time_std


class StrideWindowProcessor:
    def __init__(self, prefix_len, window_size, act_to_idx, time_mean=0.0, time_std=1.0, stride=1):
        self.case_buffer = CaseBuffer(prefix_len)
        self.prefix_len = prefix_len
        self.window_size = window_size
        self.stride = stride
        self.act_to_idx = act_to_idx
        self.pad_id = act_to_idx["<PAD>"]
        self.unk_id = act_to_idx.get("<UNK>", self.pad_id)
        self.time_mean = time_mean
        self.time_std = sanitize_time_std(time_std)
        
        self.window_queue = deque(maxlen=window_size)
        self.window_id = 0
        self.events_processed = 0
        
    def build_model_rows(self, prefix_acts, prefix_times):
        tokenized = [self.act_to_idx.get(act, self.unk_id) for act in prefix_acts]
        padded_acts = [self.pad_id] * (self.prefix_len - len(tokenized)) + tokenized
        normalized_times = build_padded_time_row(
            prefix_times, prefix_len=self.prefix_len, time_mean=self.time_mean, time_std=self.time_std
        )
        return padded_acts, normalized_times
    
    def observe_event(self, event: Event):
        """Add one event, return window once it is full"""
        self.case_buffer.add(event)
        self.events_processed += 1
        prefix_acts, prefix_times = self.case_buffer.get_prefix_for_a_case(event.case_id)
        self.window_queue.append(self.build_model_rows(prefix_acts, prefix_times))
        
        events_in_full_state = self.events_processed - self.window_size
        if events_in_full_state < 0 or events_in_full_state % self.stride != 0:
            return None
        
        self.window_id += 1
        return {
            "window_id": self.window_id,
            "X_acts": torch.tensor([acts for acts, _ in self.window_queue], dtype=torch.long),
            "X_times": torch.tensor([times for _, times in self.window_queue], dtype=torch.float32),
            "event_index": self.events_processed,
            "global_event_index": event.global_event_index,
        }
