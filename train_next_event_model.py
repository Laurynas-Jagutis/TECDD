"""TECDD Stages 1 and 2: train the transformer, then calibrate detector (PCA, R0, h, theta, theta')"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import build_activity_vocab, load_xes_traces, traces_to_replay_stream, traces_to_supervised_dataset
from dataset import make_trace_dataset
from drift_detector import DriftDetector
from main_drift import PREFIX_LEN, build_prefix_embeddings_from_stream, build_stable_window_embeddings
from models.sutran import SuTraNNextActivity, save_checkpoint
from prefix_embedder import PrefixEmbedder
from time_preprocessing import (
    TIME_PREPROCESSING_VERSION,
    fit_time_normalizer_from_padded_rows,
    infer_prefix_lengths,
    normalize_padded_time_rows,
    normalize_time_targets,
    sanitize_time_std,
)
from trainer import train_sutran_model
from window_processor import StrideWindowProcessor

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
OPENING_TRACES = 300
CALIBRATION_WINDOW_SIZE = 200
CALIBRATION_STRIDE = 18


def split_trace_list(traces, val_fraction, seed):
    val_trace_count = round(val_fraction * len(traces))
    val_indices = set(np.random.default_rng(seed).permutation(len(traces))[:val_trace_count].tolist())
    train_traces = [trace for idx, trace in enumerate(traces) if idx not in val_indices]
    val_traces = [trace for idx, trace in enumerate(traces) if idx in val_indices]
    return train_traces, val_traces


def calibrate_detector(trained_model, opening_traces, *, act_to_idx, time_mean, time_std, device):
    """Stage 2, fitting PCA, R0, h, theta and theta' on the opening region"""
    time_std = sanitize_time_std(time_std)
    stream_events = traces_to_replay_stream(opening_traces)
    embedder = PrefixEmbedder(trained_model, device=device)
    opening_embeddings = build_prefix_embeddings_from_stream(
        stream_events, PREFIX_LEN, act_to_idx, time_mean, time_std, embedder)
    processor = StrideWindowProcessor(
        PREFIX_LEN, CALIBRATION_WINDOW_SIZE, act_to_idx, time_mean, time_std, CALIBRATION_STRIDE
    )
    
    opening_windows = build_stable_window_embeddings(
        stream_events, processor, embedder, CALIBRATION_WINDOW_SIZE // CALIBRATION_STRIDE
    )
    embedder.remove_hook()
    
    detector = DriftDetector()
    detector.fit(opening_embeddings, opening_windows)
    return detector
    
    
def main():
    parser = argparse.ArgumentParser(description="Train TECDD and calibrate detector")
    parser.add_argument("--xes-path", type=Path, required=True, help="Historical (validation) log")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Where to save the checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from {args.xes_path}...", flush=True)
    traces, vocab_set = load_xes_traces(args.xes_path)
    act_to_idx, idx_to_act = build_activity_vocab(vocab_set)
    
    train_traces, val_traces = split_trace_list(traces, val_fraction=VAL_FRACTION, seed=args.seed)
    X_train, X_train_t, y_train, y_train_t = traces_to_supervised_dataset(train_traces, PREFIX_LEN, act_to_idx)
    X_val, X_val_t, y_val, y_val_t = traces_to_supervised_dataset(val_traces, PREFIX_LEN, act_to_idx)
    
    pad_id = act_to_idx["<PAD>"]
    train_prefix_lengths = infer_prefix_lengths(X_train, pad_id)
    val_prefix_lengths = infer_prefix_lengths(X_val, pad_id)
    time_mean, time_std = fit_time_normalizer_from_padded_rows(X_train_t, train_prefix_lengths)
    X_train_t = normalize_padded_time_rows(X_train_t, train_prefix_lengths, time_mean, time_std)
    X_val_t = normalize_padded_time_rows(X_val_t, val_prefix_lengths, time_mean, time_std)
    y_train_t = normalize_time_targets(y_train_t, time_mean, time_std)
    y_val_t = normalize_time_targets(y_val_t, time_mean, time_std)
    
    print(f"Traces -> Train: {len(train_traces)} | Val: {len(val_traces)}", flush=True)
    print(f"Examples -> Train: {len(X_train)} | Val: {len(X_val)} | Vocab: {len(act_to_idx)}", flush=True)
    
    train_loader = DataLoader(make_trace_dataset(X_train, X_train_t, y_train, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(make_trace_dataset(X_val, X_val_t, y_val, y_val_t), batch_size=BATCH_SIZE, shuffle=False)
         
    model = SuTraNNextActivity(vocab_size_pred=len(act_to_idx), pad_idx=pad_id, max_prefix_len=PREFIX_LEN)
    training_start_time = time.perf_counter()
    trained_model = train_sutran_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    training_runtime_seconds = time.perf_counter() - training_start_time
    print(f"\nTraining runtime: {training_runtime_seconds:.2f} s", flush=True)
    
    #stage 2: calibration
    opening_traces = traces[:OPENING_TRACES]
    print(f"\n--- Calibrating the detector on the first {len(opening_traces)} traces ---", flush=True)
    detector = calibrate_detector(
        trained_model, opening_traces, act_to_idx=act_to_idx, time_mean=time_mean, time_std=time_std, device=device
    )
    
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path=args.checkpoint_path,
        model=trained_model,
        extra={
            "source_log": str(args.xes_path),
            "act_to_idx": act_to_idx,
            "idx_to_act": idx_to_act,
            "time_mean": float(time_mean),
            "time_std": float(time_std),
            "time_preprocessing_version": TIME_PREPROCESSING_VERSION,
            "train_trace_count": len(train_traces),
            "val_trace_count": len(val_traces),
            "training_runtime_seconds": training_runtime_seconds,
            "training_device": str(device),
            "detector_state": detector.export_state(),
        },
    )
    print(f"\nSaved checkpoint to {args.checkpoint_path}", flush=True)
        
        
if __name__ == "__main__":
    main()
