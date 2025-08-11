#!/usr/bin/env python3
import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mne
import csv

from train_eegnet import _per_epoch_zscore


def load_epochs_from_edf(edf_path: str, tmin: float, tmax: float, resample_hz: int | None):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design="firwin", verbose="ERROR")
    if resample_hz is not None and abs(raw.info["sfreq"] - resample_hz) > 1e-3:
        raw.resample(resample_hz)
    events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
    if not idmap or not all(k in idmap for k in ("T1", "T2")):
        raise RuntimeError("Missing T1/T2 events")
    epochs = mne.Epochs(raw, events, event_id={"T1": idmap["T1"], "T2": idmap["T2"]}, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose="ERROR")
    x = epochs.get_data(copy=True).astype(np.float32)
    y_map = {idmap["T1"]: 0, idmap["T2"]: 1}
    y = np.array([y_map[e[-1]] for e in epochs.events], dtype=np.int64)
    x = _per_epoch_zscore(x)
    x = x[..., np.newaxis]
    return x, y


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single EDF file")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--edf", type=str, required=True)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument("--resample-hz", type=int, default=128)
    parser.add_argument("--out-csv", type=str, default="predictions.csv")
    args = parser.parse_args()

    resample = args.resample_hz if args.resample_hz > 0 else None
    x, y = load_epochs_from_edf(args.edf, args.tmin, args.tmax, resample)

    model = keras.models.load_model(args.model)
    prob = model.predict(x, verbose=0)
    pred = np.argmax(prob, axis=1)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch_index", "true_label", "pred_label", "prob_T0", "prob_T1"])  # T1=0, T2=1
        for i in range(len(pred)):
            w.writerow([i, int(y[i]), int(pred[i]), float(prob[i, 0]), float(prob[i, 1])])
    print("Saved:", args.out_csv)


if __name__ == "__main__":
    main()