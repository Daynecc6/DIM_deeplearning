# Deep InfoMax Pytorch Project

This project builds upon the work from [DuaneNielsen/DeepInfomaxPytorch](https://github.com/DuaneNielsen/DeepInfomaxPytorch).

Pytorch implementation of Deep InfoMax [https://arxiv.org/abs/1808.06670](https://arxiv.org/abs/1808.06670).

This project implements and evaluates Deep InfoMax (DIM) models using PyTorch for representation learning on the CIFAR-10 dataset. It includes scripts for training DIM encoders, training classifiers on top of these learned features, and evaluating model performance.

## Prerequisites

- Python 3.8+
- pip (Python package installer)

## Setup

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>  # Or download and extract the ZIP
    cd DeepInfomaxPytorch
    ```

    (If you've already cloned/downloaded, just navigate to the project directory)

2.  **Create and activate a virtual environment (recommended):**

    - On macOS and Linux:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Directory Structure

```
DeepInfomaxPytorch/
├── cifar/                    # CIFAR-10 dataset will be downloaded here by torchvision
├── data/                     # (May contain older dataset versions, `cifar/` is primary)
├── images/                   # (If you have local images for a README, place them here)
├── models/                   # Trained models and evaluation results are saved here
│   └── run<ID>/               # Each run_id gets its own subdirectory
│       ├── encoder<EPOCH>.wgt  # Encoder weights from train.py
│       ├── loss<EPOCH>.wgt     # DIM Loss module weights from train.py
│       ├── opt<EPOCH>.pt       # Optimizer states
│       ├── w_dim<EPOCH>.mdl    # Classifier model weights (from classification.py)
│       └── results.pt          # Evaluation results (from evaluate_models.py)
├── classification.py         # Script for training a classifier on top of DIM features
├── evaluate_models.py        # Script to evaluate trained models and generate plots
├── models.py                 # Contains model architecture definitions
├── README.md                 # This file
├── requirements.txt          # Project dependencies
└── train.py                  # Main script for training Deep InfoMax (DIM) models
```

## Running the Code

All scripts use command-line arguments for configuration. Make sure your virtual environment is activated before running any commands. The CIFAR-10 dataset will be downloaded automatically on first run if not present in the `cifar/` directory.

### 1. Training Deep InfoMax Models (`train.py`)

This script trains the DIM encoder.

**Basic Usage:**

```bash
python train.py --run_id <unique_run_id> --epochs <number_of_epochs>
```

- `--run_id`: An integer to identify the training run (e.g., 1, 2, 101). Checkpoints will be saved in `models/run<run_id>/`.
- `--epochs`: Number of epochs to train for.

**Example:**

```bash
python train.py --run_id 1 --epochs 100
```

**Advanced Options (see script help `python train.py -h` for all options):**

- **Self-Attention:**
  ```bash
  python train.py --run_id 2 --epochs 100 --self_attention --sa_channels 32
  ```
- **Patch-Only Embedding:**
  ```bash
  python train.py --run_id 3 --epochs 100 --patch_only --patch_size 4
  ```
- **Squeeze-and-Excitation (SE) Local:**
  ```bash
  python train.py --run_id 4 --epochs 100 --se_local
  ```
- **Resuming Training:**
  To resume from a specific epoch (e.g., epoch 50 of run_id 1):
  ```bash
  python train.py --run_id 1 --epochs 100 --resume 50
  ```

### 2. Training a Classifier (`classification.py`)

This script trains a linear classifier on top of the features learned by a pre-trained DIM encoder.

**Basic Usage:**

```bash
python classification.py --run_id <encoder_run_id> --encoder_epoch <encoder_epoch_to_load> --epochs <classifier_epochs>
```

- `--run_id`: The `run_id` of the pre-trained encoder (models must exist in `models/run<encoder_run_id>/`).
- `--encoder_epoch`: The epoch number of the encoder weights to load (e.g., 100, if `encoder100.wgt` exists for that run).
- `--epochs`: Number of epochs to train the classifier. The resulting classifier model (`w_dim<EPOCH>.mdl`) will be saved under `models/run<encoder_run_id>/`.

**Example:**
To train a classifier using encoder from `run_id=1` at epoch `100` for `30` epochs:

```bash
python classification.py --run_id 1 --encoder_epoch 100 --epochs 30
```

If the encoder was trained with specific flags (e.g., `--self_attention`, `--se_local`), ensure you pass those same flags to `classification.py` so it can correctly reconstruct the encoder architecture:

```bash
python classification.py --run_id 2 --encoder_epoch 100 --epochs 30 --self_attention --sa_channels 32
python classification.py --run_id 4 --encoder_epoch 100 --epochs 30 --se_local
```

**Reloading a Classifier:**
To continue training an existing classifier (e.g., reload epoch 30 and train up to epoch 60):

```bash
python classification.py --run_id 1 --encoder_epoch 100 --epochs 60 --reload 30
```

### 3. Evaluating Models (`evaluate_models.py`)

This script evaluates one or more trained classifier models and generates comparison plots.

**Usage:**

```bash
python evaluate_models.py --runs <run_id_1> <run_id_2> ...
```

- `--runs`: A list of `run_id`s. The script will look for the latest classifier model (`w_dim<EPOCH>.mdl`) in each `models/run<ID>/` directory.

**Example:**
To evaluate models from `run_id=1`, `run_id=11`, and `run_id=114`:

```bash
python evaluate_models.py --runs 1 11 114
```

The script will:

- Load the latest `.mdl` file from each specified run directory.
- Evaluate them on the CIFAR-10 test set.
- Save detailed results (including loss, accuracy, precision, recall, and confusion matrix) as `results.pt` in each run's directory.
- Generate and save comparison plots in the project's root directory:
  - `accuracy_comparison.png`
  - `precision_comparison.png`
  - `recall_comparison.png`
  - `confusion_matrix.png` (for the best performing model among those evaluated)

## Output

- **Trained Models:**
  - Encoder weights from `train.py`: `models/run<ID>/encoder<EPOCH>.wgt`
  - DIM loss module weights from `train.py`: `models/run<ID>/loss<EPOCH>.wgt`
  - Classifier models from `classification.py`: `models/run<ID>/w_dim<EPOCH>.mdl`
  - Optimizer states: `models/run<ID>/opt<EPOCH>.pt`
- **Evaluation Results:**
  - `evaluate_models.py` saves `results.pt` in each evaluated `models/run<ID>/` directory.
- **Plots:**
  - `evaluate_models.py` saves `accuracy_comparison.png`, `precision_comparison.png`, `recall_comparison.png`, and `confusion_matrix.png` in the project root.
