# Quantum-based-classification-for-DR

Hybrid quantum-classical image classifier for Diabetic Retinopathy (DR) built using TensorFlow, PennyLane and a quantum-inspired layer.

## Overview

This repository contains `qcl.py`, an end-to-end script that demonstrates a hybrid approach for classifying retinal fundus images into DR severity levels. It combines a frozen EfficientNetB0 feature extractor, PCA for dimensionality reduction, and a small quantum layer implemented with PennyLane that plugs into a TensorFlow model.

## Key Features

- Uses `EfficientNetB0` (pretrained) for feature extraction
- Applies `PCA` to reduce features to a small number of components (mapped to qubits)
- Implements a custom `QuantumLayer` using PennyLane and `default.qubit.tf`
- Trains and evaluates a hybrid model; outputs a confusion matrix, classification report, and saves the trained model

## Repository structure

- `qcl.py` — Main script that builds, trains, and evaluates the hybrid model

## Dataset expectations

The script expects a local dataset arranged like:

D:\Diabetic retinopathy\archive\colored_images\<class_folder>\*.jpg

Where `<class_folder>` names map to labels in `qcl.py`:

- `No_DR`
- `Mild`
- `Moderate`
- `Proliferate_DR`
- `Severe`

You can change the dataset location by editing the `dataset_path` and `image_dir` variables at the top of `qcl.py`.

## Requirements

- Python 3.8+ recommended
- numpy
- opencv-python
- scikit-learn
- tensorflow (tested with TF 2.x)
- pennylane
- matplotlib
- seaborn

Install dependencies with pip:

```bash
pip install numpy opencv-python scikit-learn tensorflow pennylane matplotlib seaborn
```

Note: Training a model using EfficientNet and a quantum layer can be resource intensive — a machine with a GPU is recommended.

## Usage

From the project root run:

```bash
python qcl.py
```

What the script does (high-level):

- Loads images from `colored_images` and maps folders to labels
- Normalizes and one-hot encodes labels
- Extracts features with `EfficientNetB0` (frozen)
- Applies `PCA` to reduce dimensionality to `num_qubits`
- Builds a small sequential model with the custom `QuantumLayer`
- Trains the model and evaluates on a hold-out test set
- Saves the trained model to `quantum_model.keras`

Outputs:

- Printed training progress and a final test accuracy
- A plotted confusion matrix and a printed classification report
- Saved model file: `quantum_model.keras`

## Customization

- `num_qubits`: change in `qcl.py` to adjust the PCA output size / quantum layer input
- `epochs`, `batch_size`: modify training parameters in `qcl.py`
- `EfficientNetB0` behavior: currently frozen; set `base_model.trainable = True` to fine-tune

## Troubleshooting

- If images fail to load, check file extensions and the `image_dir` path.
- For performance issues, reduce image size or number of training epochs.
- If PennyLane device errors occur, ensure compatible versions of `pennylane` and `tensorflow` are installed.


