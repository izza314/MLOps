# Fashion ANN Pipeline

End-to-End ML Versioning with Git, DVC \& Google Drive

## Project Overview

A fully-connected Artificial Neural Network (ANN) for classifying Fashion-MNIST images into 10 clothing categories.

## Technologies

* Python
* TensorFlow
* Git \& GitHub
* DVC
* Google Drive
* Scikit-learn
* Matplotlib


## Part D — DVC Pipeline

The project uses a DVC pipeline with four stages:

1. **Prepare** — downloads and stores the Fashion-MNIST raw dataset.
2. **Preprocess** — normalizes the data and creates training, validation, and test sets.
3. **Train** — trains the TensorFlow neural network using parameters from `params.yaml`.
4. **Evaluate** — evaluates the trained model and generates metrics and a confusion matrix.

The pipeline is defined in `dvc.yaml`, while `dvc.lock` records the exact dependency and output versions used for reproducibility.

The pipeline was reproduced successfully using:

```bash
dvc repro