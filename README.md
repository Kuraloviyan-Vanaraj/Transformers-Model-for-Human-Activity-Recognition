# Transformer Model for Human Activity Recognition

A Master's project investigating **Transformer-based human activity recognition (HAR)** from wearable inertial sensor data.

The project uses the inertial modality of the **UTD-MHAD (UTD Multimodal Human Action Dataset)** and includes a main train/test experiment, repeated runs, and 5-fold cross-validation.

> **Dataset notice:** The UTD-MHAD data is intentionally **not included in this repository**. Download it separately from the official dataset page and place the inertial `.mat` files under `data/raw/inertial/`.

## Highlights

- Transformer-based sequence classification for inertial sensor time series
- 27 human activity classes
- Fixed-length sequence preparation with padding/truncation
- Feature standardization with `StandardScaler`
- Gaussian-noise augmentation in the experimental workflow
- Multi-head self-attention with residual connections
- Adam optimization, early stopping, and learning-rate scheduling
- Accuracy, confusion matrix, and precision evaluation
- Repeated experiments and stratified 5-fold cross-validation

## Dataset

The project uses **UTD-MHAD**, created by Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz at the University of Texas at Dallas. The official project page describes 27 actions performed by 8 subjects and 861 usable sequences after three corrupted sequences were removed.

### Download separately

1. Open the official UTD-MHAD page: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
2. Download **`Inertial_Data.zip`**.
3. Extract the `.mat` files into:

```text
data/raw/inertial/
```

See `data/README.md` for the dataset citation and setup details.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── train.py
│   ├── repeated_runs.py
│   └── cross_validation.py
├── docs/
│   └── PROJECT_NOTES.md
└── results/
    └── README.md
```

The original notebooks from the Master's project were refactored into importable Python modules so the public repository is easier to maintain and reproduce.

## Methodology

### 1. Data loading

MATLAB inertial recordings are loaded with `scipy.io.loadmat`. Each filename encodes the action, subject, and trial.

### 2. Sequence preparation

Sequences are padded or truncated to a maximum of 200 time steps.

### 3. Normalization

Each sequence is flattened temporarily and standardized with `StandardScaler`, then reshaped for the Transformer.

### 4. Transformer model

The model uses stacked multi-head self-attention blocks with batch normalization, ReLU activations, residual connections, global average pooling, dense layers, dropout, and a softmax classification head.

### 5. Experiments

- `src/train.py` — main train/test experiment
- `src/repeated_runs.py` — repeated train/test runs
- `src/cross_validation.py` — stratified 5-fold cross-validation

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

Run everything from the repository root after downloading the dataset:

```bash
python -m src.train
python -m src.repeated_runs
python -m src.cross_validation
```

Generated model files and experiment artifacts belong under `results/` and are ignored by Git by default.

## Reported project results

The original Master's report documents these historical results:

| Evaluation | Accuracy | Weighted precision |
|---|---:|---:|
| Standard train/test evaluation | 85.71% | 0.7648 |
| 5-fold cross-validation | 92% average | 0.7548 |

These are **historical results from the original project**, not newly reproduced benchmark measurements. Re-running the experiments may produce different results because of software versions, random seeds, hardware, preprocessing, and split details.

## Reproducibility notes

Before treating the project as a research benchmark, verify:

- exact train/test split protocol
- whether evaluation is subject-independent
- preprocessing and scaler fitting order
- random seeds and deterministic settings
- TensorFlow/CUDA compatibility
- dataset version and file integrity

## Future work

Potential extensions include subject-independent evaluation, CNN/LSTM/GRU baselines, systematic hyperparameter tuning, experiment tracking, attention visualization, and multimodal sensor fusion.

## Dataset citation

Chen, C., Jafari, R., & Kehtarnavaz, N. (2015). *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor*. Proceedings of the IEEE International Conference on Image Processing (ICIP).

## Model reference

Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems.
