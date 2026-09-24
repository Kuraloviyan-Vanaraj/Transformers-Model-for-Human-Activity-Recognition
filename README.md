# Transformer Model for Human Activity Recognition

A Master's project investigating **Transformer-based human activity recognition (HAR)** from wearable inertial sensor data.

The experiments use the inertial modality of the **UTD-MHAD (UTD Multimodal Human Action Dataset)** and evaluate a Transformer sequence-classification model across a standard experiment, repeated runs, and 5-fold cross-validation.

> **Dataset notice:** The UTD-MHAD data is intentionally **not included in this repository**. Download it separately from the official dataset page and place the inertial `.mat` files under `data/raw/inertial/`.

## What this project does

The project treats inertial-sensor recordings as multivariate time series and uses a Transformer architecture to learn temporal dependencies for multi-class activity recognition.

The original workflow includes:

- sequence padding/truncation to a fixed length
- feature standardization with `StandardScaler`
- multi-class label encoding
- noise-based augmentation in the experimental workflow
- Transformer self-attention and positional information
- Adam optimization
- early stopping and learning-rate scheduling
- confusion-matrix and per-class precision analysis
- repeated training runs
- stratified 5-fold cross-validation

## Dataset

The project uses the **UTD-MHAD** dataset created by Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz at the University of Texas at Dallas. The official dataset page describes 27 actions performed by 8 subjects and reports 861 usable sequences after three corrupted sequences were removed. The inertial sensor records acceleration and angular-velocity signals.

### Download the dataset separately

1. Open the official UTD-MHAD page: https://personal.utdallas.edu/~kehtar/UTD-MHAD.html
2. Download **`Inertial_Data.zip`** from the Download section.
3. Extract the `.mat` files into:

```text
data/raw/inertial/
```

The repository's `data/README.md` contains the same instructions and the dataset citation.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_transformer_activity_recognition.ipynb
│   ├── 02_repeated_runs.ipynb
│   └── 03_five_fold_cross_validation.ipynb
├── docs/
│   └── PROJECT_NOTES.md
└── results/
    └── README.md
```

## Activity classes

UTD-MHAD contains 27 action classes, including swipes, waving, clapping, throwing, drawing gestures, sports movements, boxing, walking/jogging, sit-to-stand, stand-to-sit, lunging, and squatting. The official page provides the complete numbered list.

## Methodology

### 1. Data loading

The notebooks load MATLAB inertial recordings with `scipy.io.loadmat` and extract the inertial signal used for classification.

### 2. Sequence preparation

Sequences are converted to a fixed maximum length of 200 time steps through padding and truncation, matching the original project workflow.

### 3. Normalization

Sensor features are standardized using `StandardScaler` before being passed to the model.

### 4. Transformer model

The model uses the standard Transformer building blocks: multi-head self-attention, feed-forward transformations, residual connections, normalization, dropout, and a final softmax classification layer.

### 5. Evaluation

The experiments report accuracy, confusion matrices, per-class precision, and weighted precision. The repository also preserves the repeated-run and 5-fold cross-validation workflows from the Master's project.

## Reported project results

The original Master's report documents the following historical results:

| Evaluation | Accuracy | Weighted precision |
|---|---:|---:|
| Standard train/test evaluation | 85.71% | 0.7648 |
| 5-fold cross-validation | 92% average | 0.7548 |

These are **results reported by the original project**, not a newly reproduced benchmark. Re-running the notebooks can produce different values depending on software versions, random seeds, hardware, preprocessing, and split details.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate

# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter from the repository root:

```bash
jupyter notebook
```

## Run the experiments

Make sure the dataset has been extracted into `data/raw/inertial/`, then run the notebooks:

```text
01_transformer_activity_recognition.ipynb
02_repeated_runs.ipynb
03_five_fold_cross_validation.ipynb
```

Generated plots, CSV files, model checkpoints, and other experiment artifacts can be stored under `results/` and are ignored by Git by default.

## Reproducibility notes

This repository preserves the original Master's-project notebooks rather than turning them into a production ML package. Before using the project as a research benchmark, verify:

- random seeds and deterministic settings
- the exact train/test split protocol
- whether the split is subject-independent
- preprocessing order and fitted scalers
- TensorFlow and CUDA/cuDNN compatibility
- dataset version and file integrity

## Future work

Potential extensions include subject-independent evaluation, stronger experiment tracking, baseline comparisons against CNN/LSTM/GRU models, hyperparameter tuning, attention visualization, and multimodal fusion.

## Dataset citation

Chen, C., Jafari, R., & Kehtarnavaz, N. (2015). *UTD-MHAD: A Multimodal Dataset for Human Action Recognition Utilizing a Depth Camera and a Wearable Inertial Sensor*. Proceedings of the IEEE International Conference on Image Processing (ICIP).

## Model reference

Vaswani, A. et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems.
