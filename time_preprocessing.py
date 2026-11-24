import numpy as np

TIME_PAD_VALUE = 0.0
TIME_PREPROCESSING_VERSION = 1


def sanitize_time_std(time_std):
    std = float(time_std)
    return std if np.isfinite(std) and std != 0.0 else 1.0


def infer_prefix_lengths(X_acts, pad_idx):
    """# of non padded positions of every left padded row"""
    return np.sum(np.asarray(X_acts) != pad_idx, axis=1, dtype=np.int64)


def fit_time_normalizer_from_padded_rows(X_times, prefix_lengths):
    """Mean and std of the time gaps at real positions only"""
    time_rows = np.asarray(X_times, dtype=np.float32)
    observed = [row[-length:] for row, length in zip(time_rows, prefix_lengths) if length > 0]
    if not observed:
        return 0.0, 1.0
    observed = np.concatenate(observed).astype(np.float64)
    time_mean = float(np.mean(observed))
    return(time_mean if np.isfinite(time_mean) else 0.0), sanitize_time_std(np.std(observed))


def normalize_time_targets(y_times, time_mean, time_std):
    targets = np.asarray(y_times, dtype=np.float32)
    return ((targets - float(time_mean)) / sanitize_time_std(time_std)).astype(np.float32)


def normalize_padded_time_rows(X_times, prefix_lengths, time_mean, time_std):
    time_rows = np.asarray(X_times, dtype=np.float32)
    normalized = np.full(time_rows.shape, TIME_PAD_VALUE, dtype=np.float32)
    safe_std = sanitize_time_std(time_std)
    for row_idx, length in enumerate(prefix_lengths):
        if length > 0:
            normalized[row_idx, -length:] = (time_rows[row_idx, -length:] - float(time_mean)) / safe_std
    return normalized


def build_padded_time_row(prefix_times, prefix_len, time_mean, time_std):
    safe_std = sanitize_time_std(time_std)
    normalized = [float((float(value) - float(time_mean)) / safe_std) for value in prefix_times]
    return [TIME_PAD_VALUE] * (prefix_len - len(normalized)) + normalized
