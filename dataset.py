import torch
from torch.utils.data import TensorDataset


def make_trace_dataset(X_acts, X_times, y_acts, y_times):
    """(prefix activities, prefix time gaps, next activity, time until it) per example"""
    return TensorDataset(
        torch.as_tensor(X_acts, dtype=torch.long),
        torch.as_tensor(X_times, dtype=torch.float32),
        torch.as_tensor(y_acts, dtype=torch.long),
        torch.as_tensor(y_times, dtype=torch.float32)
    )
