#!/usr/bin/env python3
import argparse
import os
from typing import Dict, Tuple

import numpy as np
import mne
from tqdm.auto import tqdm

# Keep in sync with train_eegnet defaults
LOW_FREQ_HZ = 1.0
HIGH_FREQ_HZ = 40.0
EPOCH_TMIN_S = 0.0
EPOCH_TMAX_S = 4.0
RESAMPLE_HZ = 128


def process_file(edf_path: str, tmin: float, tmax: float, resample_hz: int | None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    if raw.info["nchan"] == 0:
        raise RuntimeError("No EEG channels in file")
    raw.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design="firwin", verbose="ERROR")
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-3:
        raw.resample(resample_hz)
    events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
    if not idmap or not all(k in idmap for k in ("T1", "T2")):
        raise RuntimeError("Missing T1/T2 events")
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"T1": idmap["T1"], "T2": idmap["T2"]},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    if len(epochs) == 0:
        raise RuntimeError("No epochs")
    x = epochs.get_data(copy=True).astype(np.float32)  # (n, C, T)
    code_to_label = {idmap["T1"]: 0, idmap["T2"]: 1}
    y = np.array([code_to_label[e[-1]] for e in epochs.events], dtype=np.int64)
    # Per-epoch z-score across time
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True) + 1e-8
    x = (x - mean) / std
    # add last dimension for channels-last
    x = x[..., np.newaxis]
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "channel_names": [ch["ch_name"] for ch in raw.info["chs"]],
        "tmin": float(tmin),
        "tmax": float(tmax),
        "resample_hz": int(resample_hz) if resample_hz is not None else None,
    }
    return x, y, meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess EEG EDFs to cached NPZ epochs")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "physionet_raw"))
    parser.add_argument("--out-root", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
    parser.add_argument("--tmin", type=float, default=EPOCH_TMIN_S)
    parser.add_argument("--tmax", type=float, default=EPOCH_TMAX_S)
    parser.add_argument("--resample-hz", type=int, default=RESAMPLE_HZ)
    args = parser.parse_args()

    data_root = args.data_root
    out_root = args.out_root
    resample = args.resample_hz if args.resample_hz > 0 else None

    # Glob S*/S*R*.edf
    subjects = sorted([d for d in os.listdir(data_root) if d.startswith("S") and os.path.isdir(os.path.join(data_root, d))])
    total = 0
    for subj in tqdm(subjects, desc="Subjects"):
        subj_dir = os.path.join(data_root, subj)
        edfs = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        for edf in tqdm(edfs, leave=False, desc=subj):
            edf_path = os.path.join(subj_dir, edf)
            rel_dir = os.path.join(out_root, subj)
            os.makedirs(rel_dir, exist_ok=True)
            base = os.path.splitext(edf)[0]
            out_path = os.path.join(rel_dir, f"{base}.npz")
            if os.path.exists(out_path):
                continue
            try:
                x, y, meta = process_file(edf_path, args.tmin, args.tmax, resample)
                np.savez_compressed(out_path, X=x, y=y, **meta)
                total += x.shape[0]
            except Exception:
                continue
    print(f"Saved cached epochs to {out_root}. Total epochs: {total}")


if __name__ == "__main__":
    main()