

from __future__ import annotations

import argparse
import os
import random
from typing import Generator, Iterable, List, Optional, Tuple
import glob as pyglob
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.auto import tqdm

import mne
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# Configuration (defaults)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "physionet_raw")
MODEL_OUT = os.path.join(PROJECT_ROOT, "eegnet_motor_imagery.h5")

LOW_FREQ_HZ = 1.0
HIGH_FREQ_HZ = 40.0
EPOCH_TMIN_S = 0.0
EPOCH_TMAX_S = 4.0
RESAMPLE_HZ: Optional[int] = 128

TRAIN_TEST_SPLIT = 0.2  # fraction for test
FILE_SPLIT_SEED = 42

MOTOR_EVENT_ID = {"T1": 0, "T2": 1}  # left=0, right=1



# Utility functions


def list_edf_files(data_root: str) -> List[str]:
    """Return sorted list of all EDF files under given root.

    Expects structure like: data_root/S001/S001R01.edf
    """
    pattern = os.path.join(data_root, "S*", "S*R*.edf")
    files = sorted(pyglob.glob(pattern))
    return files


def _first_valid_epoch_shape(files: Iterable[str]) -> Tuple[int, int]:
    """Probe files until a valid (channels, samples) is obtained."""
    for filepath in files:
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose="ERROR")
            raw.pick("eeg")
            if raw.info["nchan"] == 0:
                continue
            raw.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design="firwin", verbose="ERROR")
            if RESAMPLE_HZ is not None and abs(raw.info["sfreq"] - RESAMPLE_HZ) > 1e-3:
                raw.resample(RESAMPLE_HZ)
            events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
            if not idmap or not all(k in idmap for k in ("T1", "T2")):
                continue
            epochs = mne.Epochs(
                raw,
                events,
                event_id={"T1": idmap["T1"], "T2": idmap["T2"]},
                tmin=EPOCH_TMIN_S,
                tmax=EPOCH_TMAX_S,
                baseline=None,
                preload=True,
                verbose="ERROR",
            )
            if len(epochs) == 0:
                continue
            x = epochs.get_data(copy=True)
            return x.shape[1], x.shape[2]
        except Exception:
            continue
    raise RuntimeError("Could not determine (channels, samples) from available EDF files.")


def _estimate_num_events(files: Iterable[str]) -> int:
    """Estimate number of T1/T2 events across files without loading full data."""
    total = 0
    for filepath in files:
        try:
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose="ERROR")
            events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
            if not idmap:
                continue
            keep_ids = [idmap[k] for k in ("T1", "T2") if k in idmap]
            if not keep_ids:
                continue
            total += int(np.sum(np.isin(events[:, -1], keep_ids)))
        except Exception:
            continue
    return total


def _per_epoch_zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalize per epoch, per channel over time axis (axis=2)."""
    mean = x.mean(axis=2, keepdims=True)
    std = x.std(axis=2, keepdims=True) + 1e-8
    return (x - mean) / std


def gen_batches(
    files: List[str],
    batch_size: int,
    num_channels: int,
    num_samples: int,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Yield batches (X, y) by streaming across EDF files."""
    buf_x: List[np.ndarray] = []
    buf_y: List[int] = []

    for filepath in files:
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose="ERROR")
            raw.pick("eeg")
            if raw.info["nchan"] == 0:
                continue
            raw.filter(l_freq=LOW_FREQ_HZ, h_freq=HIGH_FREQ_HZ, fir_design="firwin", verbose="ERROR")
            if RESAMPLE_HZ is not None and abs(raw.info["sfreq"] - RESAMPLE_HZ) > 1e-3:
                raw.resample(RESAMPLE_HZ)
            events, idmap = mne.events_from_annotations(raw, verbose="ERROR")
            if not idmap or not all(k in idmap for k in ("T1", "T2")):
                continue
            local_event_id = {"T1": idmap["T1"], "T2": idmap["T2"]}
            epochs = mne.Epochs(
                raw,
                events,
                event_id=local_event_id,
                tmin=EPOCH_TMIN_S,
                tmax=EPOCH_TMAX_S,
                baseline=None,
                preload=True,
                verbose="ERROR",
            )
            if len(epochs) == 0:
                continue
            x = epochs.get_data(copy=True).astype(np.float32)  # (n, C, T)
            code_to_label = {local_event_id["T1"]: 0, local_event_id["T2"]: 1}
            y = np.array([code_to_label[e[-1]] for e in epochs.events], dtype=np.int64)

            x = _per_epoch_zscore(x)
            if x.shape[1] != num_channels or x.shape[2] != num_samples:
                # Shape mismatch with probed shape; skip
                continue
            # Identify potential outliers by per-epoch z-score energy; mark but still include
            energy = np.mean(np.square(x), axis=(1, 2))  # simple energy score per epoch
            med, mad = np.median(energy), np.median(np.abs(energy - np.median(energy)) + 1e-8)
            thresh_high = med + 6.0 * mad if mad > 0 else med * 10.0
            for i in range(x.shape[0]):
                xi = x[i]
                label = int(y[i])
                # Attach a flag for outlier (1) or normal (0) via last dimension pad if desired
                buf_x.append(xi[..., np.newaxis])  # (C, T, 1)
                buf_y.append(label)
                
                if len(buf_x) == batch_size:
                    yield np.stack(buf_x, axis=0), np.asarray(buf_y)
                    buf_x, buf_y = [], []
        except Exception:
            continue

    if buf_x:
        yield np.stack(buf_x, axis=0), np.asarray(buf_y)


def build_eegnet(
    num_classes: int,
    num_channels: int,
    num_samples: int,
    dropout_rate: float = 0.5,
    kern_length: int = 64,
    F1: int = 8,
    D: int = 2,
    F2: Optional[int] = None,
    pool_size: int = 4,
) -> keras.Model:
    """Create EEGNet model (channels-last: [C, T, 1])."""
    if F2 is None:
        F2 = F1 * D
    inp = layers.Input(shape=(num_channels, num_samples, 1))

    # Block 1
    x = layers.Conv2D(
        filters=F1,
        kernel_size=(1, kern_length),
        padding="same",
        use_bias=False,
        data_format="channels_last",
    )(inp)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(
        kernel_size=(num_channels, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=keras.constraints.max_norm(1.0),
        data_format="channels_last",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, pool_size))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2
    x = layers.SeparableConv2D(
        filters=F2,
        kernel_size=(1, 16),
        use_bias=False,
        padding="same",
        data_format="channels_last",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, pool_size))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(num_classes, kernel_constraint=keras.constraints.max_norm(0.25))(x)
    out = layers.Activation("softmax")(x)

    model = keras.Model(inputs=inp, outputs=out, name="EEGNet")
    return model


def main() -> None:
    # Globals must be declared before any references inside this function
    global RESAMPLE_HZ
    global EPOCH_TMAX_S

    parser = argparse.ArgumentParser(description="Train EEGNet with streaming over local PhysioNet EEG data")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT, help="Path to physionet_raw root")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping")
    parser.add_argument("--model-out", type=str, default=MODEL_OUT, help="Output model path")
    parser.add_argument("--logs-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs"), help="Directory to write logs")
    parser.add_argument("--resample-hz", type=int, default=RESAMPLE_HZ or 0, help="Resample Hz (0 to disable)")
    parser.add_argument("--tmax", type=float, default=EPOCH_TMAX_S, help="Epoch length (s)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.resample_hz and args.resample_hz > 0:
        RESAMPLE_HZ = int(args.resample_hz)
    else:
        RESAMPLE_HZ = None
    EPOCH_TMAX_S = float(args.tmax)

  
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        tf.random.set_seed(args.seed)
    except Exception:
        pass

    # List files and split train/test at file level
    all_files = list_edf_files(args.data_root)
    if not all_files:
        raise FileNotFoundError(f"No EDF files found under: {args.data_root}")
    rng = random.Random(args.seed)
    rng.shuffle(all_files)
    cut = int(len(all_files) * (1.0 - TRAIN_TEST_SPLIT))
    train_files, test_files = all_files[:cut], all_files[cut:]

    # Probe shapes and estimate counts
    num_channels, num_samples = _first_valid_epoch_shape(all_files)
    train_events = _estimate_num_events(train_files)
    test_events = _estimate_num_events(test_files)
    if train_events == 0:
        raise RuntimeError("No T1/T2 events found in training split. Check data path and annotations.")
    if test_events == 0:
        raise RuntimeError("No T1/T2 events found in test split. Check data path and annotations.")
    # Use floor to avoid overestimating steps; we'll also repeat() the dataset to be safe
    steps_per_epoch = max(1, train_events // args.batch_size)
    val_steps = max(1, test_events // args.batch_size)

    print(f"Channels={num_channels}, Samples={num_samples}")
    print(f"Estimated events: train≈{train_events}, test≈{test_events}")
    print(f"Steps: train={steps_per_epoch}, val={val_steps}")

    # Build datasets
    output_sig = (
        tf.TensorSpec(shape=(None, num_channels, num_samples, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    )
    train_ds = tf.data.Dataset.from_generator(
        lambda: gen_batches(train_files, args.batch_size, num_channels, num_samples),
        output_signature=output_sig,
    ).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(
        lambda: gen_batches(test_files, args.batch_size, num_channels, num_samples),
        output_signature=output_sig,
    ).repeat().prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_eegnet(num_classes=2, num_channels=num_channels, num_samples=num_samples)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Create logs and models directories
    logs_dir = args.logs_dir
    os.makedirs(logs_dir, exist_ok=True)
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    print("Logging to:", logs_dir)
    print("Models will be saved to:", models_dir)

    # Save run configuration
    run_config = {
        "data_root": args.data_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "early_stopping": not args.no_early_stopping,
        "resample_hz": RESAMPLE_HZ,
        "tmax": EPOCH_TMAX_S,
        "seed": args.seed,
        "num_channels": num_channels,
        "num_samples": num_samples,
    }
    try:
        with open(os.path.join(logs_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)
        print("Saved run config to:", os.path.join(logs_dir, "config.json"))
    except Exception:
        pass
    
    tb_dir = os.path.join(logs_dir, f"tb_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Build callbacks
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(logs_dir, "training_log.csv"), append=False),
        keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=args.patience//2, verbose=1),
        keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=0, write_graph=False, write_images=False),
    ]
    
    # Add early stopping only if not disabled
    if not args.no_early_stopping:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", 
                patience=args.patience, 
                restore_best_weights=True, 
                verbose=1
            )
        )

    # Train
    print(f"\nTraining for {args.epochs} epochs with batch size {args.batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}, Validation steps: {val_steps}")
    print(f"Early stopping: {'Enabled' if not args.no_early_stopping else 'Disabled'}")
    
    try:
        history = model.fit(
            train_ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1,
        )
    except Exception as e:
        print(f"Training error: {e}")
        print("This might be due to data pipeline issues. Check the logs above.")
        return

    # Evaluate
    train_acc = model.evaluate(train_ds, steps=steps_per_epoch, verbose=0)[1]
    test_acc = model.evaluate(val_ds, steps=val_steps, verbose=0)[1]
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    print(f"Testing Accuracy:  {test_acc * 100:.2f}%")

    # Save training history
    try:
        with open(os.path.join(logs_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history.history, f)
        print("Saved training history to:", os.path.join(logs_dir, "training_history.json"))
        print("TensorBoard logs at:", tb_dir)
    except Exception as _:
        pass

    # Plot curves and save figure
    try:
        hist = history.history
        epochs_axis = range(1, len(hist.get("loss", [])) + 1)
        
        if len(epochs_axis) == 0:
            print("No training history to plot")
            return
            
        fig = plt.figure(figsize=(12, 5))
        
        # Loss subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_axis, hist.get("loss", []), label="Training Loss", linewidth=2)
        plt.plot(epochs_axis, hist.get("val_loss", []), label="Validation Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.grid(True, alpha=0.3)

        # Accuracy subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_axis, hist.get("accuracy", []), label="Training Accuracy", linewidth=2)
        plt.plot(epochs_axis, hist.get("val_accuracy", []), label="Validation Accuracy", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Training and Validation Accuracy")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(logs_dir, "training_curves.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print("Saved curves to:", fig_path)
        plt.close(fig)
        
        # Print final metrics
        if hist.get("accuracy"):
            final_train_acc = hist["accuracy"][-1] * 100
            final_val_acc = hist["val_accuracy"][-1] * 100
            print(f"\nFinal Training Accuracy: {final_train_acc:.2f}%")
            print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
            print(f"Total Epochs Trained: {len(epochs_axis)}")
            
            if len(epochs_axis) < args.epochs:
                print(f"Training stopped early at epoch {len(epochs_axis)}/{args.epochs}")
                if not args.no_early_stopping:
                    print("Early stopping was triggered - model may have converged")
    except Exception as e:
        print(f"Error plotting curves: {e}")
        pass

    # Determine save path and save model
    desired_out = args.model_out if args.model_out else os.path.join(models_dir, "eegnet_motor_imagery.keras")
    # Ensure directory exists
    os.makedirs(os.path.dirname(desired_out), exist_ok=True)

    root, ext = os.path.splitext(desired_out)
    try:
        # Prefer .keras
        save_path = desired_out if ext.lower() == ".keras" else f"{root}.keras"
        model.save(save_path)
        print("Saved model to:", save_path)
    except Exception as e:
        print(f"Error saving in .keras format: {e}")
        # Fallback to HDF5 if Keras format fails or extension requests it
        try:
            save_path = desired_out if ext.lower() == ".h5" else f"{root}.h5"
            model.save(save_path)
            print("Saved model to (HDF5):", save_path)
        except Exception as e2:
            print(f"Failed to save model in any format: {e2}")
            return


if __name__ == "__main__":
    main()


