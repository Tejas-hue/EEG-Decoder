# Model Card: EEGNet for Motor Imagery (T1 vs T2)

- Architecture: EEGNet (Keras implementation)
- Input: channels-last (C, T, 1) epochs; default C inferred from data, T from 0–4 s at 128 Hz
- Task: Binary classification T1 vs T2 events from EEGMMI dataset
- Training: Adam (1e-3), early stopping on val_accuracy, ReduceLROnPlateau
- Preprocessing: 1–40 Hz FIR, optional resample to 128 Hz, epoching [0, 4] s, per-epoch per-channel z-scoring

## Data
- Source: PhysioNet EEG Motor Movement/Imagery (`https://physionet.org/content/eegmmidb/`), 109 subjects
- Split: File-level 80/20

## Metrics
- Reported in `logs/eval_report.txt` via `scripts/evaluate.py` (accuracy, precision/recall/F1)

## Intended Use
- Research and educational purposes for EEG decoding; not for clinical use.

## Limitations
- Binary labels only (T1/T2); performance may vary across subjects.
- Sensitive to preprocessing choices (filter band, resampling, tmin/tmax).

## Ethical Considerations
- Respect dataset license and subject privacy; do not attempt re-identification.