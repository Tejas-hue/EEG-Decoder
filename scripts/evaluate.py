#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path for direct script execution
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_eegnet import list_edf_files, _first_valid_epoch_shape, _estimate_num_events, gen_batches, TRAIN_TEST_SPLIT


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved EEGNet model on the PhysioNet test split")
    parser.add_argument("--data-root", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "physionet_raw"))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"))
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    files = list_edf_files(args.data_root)
    random.Random(args.seed).shuffle(files)
    cut = int(len(files) * (1.0 - TRAIN_TEST_SPLIT))
    _, test_files = files[:cut], files[cut:]

    num_channels, num_samples = _first_valid_epoch_shape(files)
    test_events = _estimate_num_events(test_files)
    steps = max(1, test_events // args.batch_size)

    ds = tf.data.Dataset.from_generator(
        lambda: gen_batches(test_files, args.batch_size, num_channels, num_samples),
        output_signature=(
            tf.TensorSpec(shape=(None, num_channels, num_samples, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
        ),
    ).take(steps)

    model = keras.models.load_model(args.model)

    y_true = []
    y_pred = []
    y_prob = []
    for xb, yb in ds:
        prob = model.predict(xb, verbose=0)
        pred = np.argmax(prob, axis=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob[:, 1].tolist())

    rep = classification_report(y_true, y_pred, target_names=["T1", "T2"], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(rep)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print("Saved:", report_path)

    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", cm_path)


if __name__ == "__main__":
    main()