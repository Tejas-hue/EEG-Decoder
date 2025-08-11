#!/usr/bin/env python3
import argparse
import os
import mne
import numpy as np
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Validate PhysioNet EEGMMI data presence and event counts")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "physionet_raw"))
    args = parser.parse_args()

    subjects = sorted([d for d in os.listdir(args.data_root) if d.startswith("S") and os.path.isdir(os.path.join(args.data_root, d))])
    total_files = 0
    total_events = 0
    missing = []

    for subj in tqdm(subjects, desc="Subjects"):
        subj_dir = os.path.join(args.data_root, subj)
        edfs = sorted([f for f in os.listdir(subj_dir) if f.lower().endswith(".edf")])
        if not edfs:
            missing.append(subj)
            continue
        for edf in edfs:
            total_files += 1
            path = os.path.join(subj_dir, edf)
            try:
                raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
                events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
                if not idmap:
                    continue
                keep_ids = [idmap[k] for k in ("T1", "T2") if k in idmap]
                total_events += int(np.sum(np.isin(events[:, -1], keep_ids)))
            except Exception:
                continue

    print(f"Subjects found: {len(subjects)}")
    print(f"EDF files found: {total_files}")
    print(f"Total T1/T2 events: {total_events}")
    if missing:
        print("Subjects missing EDFs:", ", ".join(missing))


if __name__ == "__main__":
    main()