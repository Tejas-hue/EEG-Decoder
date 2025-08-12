# EEGNet Motor Imagery Training

Train an EEGNet classifier on the PhysioNet EEG Motor Movement/Imagery dataset using MNE + TensorFlow.

## Quickstart

- Create environment and install deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Download dataset to `data/physionet_raw`:

```bash
python scripts/download_data.py
```

- Train:

```bash
python train_eegnet.py --data-root data/physionet_raw --epochs 50 --batch-size 64 --resample-hz 128 --tmax 4.0
```

- Outputs:
  - Logs: `logs/`, TensorBoard subdir per run
  - Curves: `logs/training_curves.png`
  - Model: defaults to `models/eegnet_motor_imagery.keras` (use `--model-out` to override)

## Dataset

- Source: PhysioNet EEG Motor Movement/Imagery (EEGMMI)
- Download: `https://physionet.org/content/eegmmidb/get-zip/1.0.0/`
- Subjects: 109
- Expected layout after download: `data/physionet_raw/Sxxx/SxxxRyy.edf`

Acknowledgement:

EEG Motor Movement/Imagery Dataset (Sept. 9, 2009, midnight)

A set of 64-channel EEGs from subjects who performed a series of motor/imagery tasks has been contributed to PhysioNet by the developers of the BCI2000 instrumentation system for brain-computer interface research.

When using this resource, please cite the original publication:

Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE Transactions on Biomedical Engineering 51(6):1034-1043, 2004.

Please include the standard citation for PhysioNet:

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345.

## Preprocessing (optional cache)

Create cached epochs to speed up training:

```bash
python scripts/preprocess_eeg.py --data-root data/physionet_raw --out-root data/processed --resample-hz 128 --tmin 0.0 --tmax 4.0
```

## Evaluation

```bash
python scripts/evaluate.py --data-root data/physionet_raw --model models/eegnet_motor_imagery.keras --batch-size 128 --seed 123 --out-dir logs
```

Example results (test split):

```
              precision    recall  f1-score   support

          T1     0.7496    0.7452    0.7474      2045
          T2     0.7467    0.7511    0.7489      2045

    accuracy                         0.7482      4090
   macro avg     0.7482    0.7482    0.7482      4090
weighted avg     0.7482    0.7482    0.7482      4090
```

## Inference on single EDF

```bash
python scripts/infer.py --model models/eegnet_motor_imagery.keras --edf data/physionet_raw/S001/S001R01.edf --out-csv logs/preds.csv
```

## Notebooks

- `notebooks/01_explore_data.ipynb` — basic EDA
- `notebooks/02_train_eegnet.ipynb` — training demo
- `notebooks/03_evaluate_and_visualize.ipynb` — evaluation + confusion matrix

## CI, LFS, Docker, and Dev

- CI: GitHub Actions workflow in `.github/workflows/ci.yml` runs tests on push/PR.
- LFS: `.gitattributes` defines large artifacts (models, npz, edf, png). Run `git lfs install` before committing large files.
- Docker: `Dockerfile` provides a reproducible environment. Build with `docker build -t eegnet .`.
- Pre-commit: install hooks with `pip install pre-commit && pre-commit install`.

## Notes

- Data, logs and models are gitignored by default.
- You can point `--data-root` anywhere that matches `S*/S*R*.edf`.
- See `DATASET.md` for details and citations, and `MODEL_CARD.md` for model info.
