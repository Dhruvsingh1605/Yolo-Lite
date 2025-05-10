# YOLOv4-Tiny Furniture Detection with MLOps

This repository implements a domain-specific furniture detection system using YOLOv4-Tiny in TensorFlow, complete with an end-to-end MLOps pipeline. It features data conversion, model training via Darknet, evaluation, TensorFlow/TFLite export, and real-time webcam inference—all orchestrated with Git, DVC, and MLflow for full reproducibility and traceability.

## Table of Contents

* [Project Overview](#project-overview)
* [Directory Structure](#directory-structure)
* [Setup Instructions](#setup-instructions)
* [Data Preparation](#data-preparation)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [Model Export](#model-export)
* [Real-Time Inference](#real-time-inference)
* [MLOps Pipeline](#mlops-pipeline)
* [Experiment Tracking](#experiment-tracking)
* [Versioning and Reproduction](#versioning-and-reproduction)
* [Contributing](#contributing)
* [License](#license)

## Project Overview

A real-time furniture detection solution built on YOLOv4-Tiny optimized for CPU-only environments (Intel i7, 8 GB RAM). The pipeline:

1. Converts Roboflow YOLOv8 dataset to Darknet format
2. Trains a YOLOv4-Tiny model via Darknet
3. Evaluates model performance (mAP, loss)
4. Exports to TensorFlow SavedModel & TFLite
5. Runs real-time webcam detection with OpenCV and TFLite

All steps are versioned with DVC, tracked with Git, and experiments logged in MLflow.

## Directory Structure

```
project_root/
├── .gitignore
├── dvc.yaml
├── params.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── yolov4-tiny/
├── logs/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export.py
│   └── detect.py
├── tensorflow-yolov4-tflite/
└── mlruns/
```

## Setup Instructions

1. **Clone the repo**:

   ```bash
   git clone <repo_url>
   cd project_root
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Initialize DVC & MLflow**:

   ```bash
   dvc init
   mlflow ui  # runs at http://localhost:5000
   ```

## Data Preparation

Run:

```bash
python scripts/prepare_data.py
```

* Reads `data/raw/data.yaml`
* Generates `data/processed/classes.names`, `train.txt`, `val.txt`
* Copies images and labels to `data/processed/`

## Model Training

Run:

```bash
python scripts/train.py
```

* Launches Darknet training of YOLOv4-Tiny
* Logs hyperparameters and metrics to MLflow
* Saves final weights to `models/yolov4-tiny/last.weights`

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

* Computes mAP on validation set
* Logs evaluation metrics and artifacts

## Model Export

Run:

```bash
python scripts/export.py
```

* Converts Darknet weights → TensorFlow SavedModel
* Exports to `models/yolov4-tiny/saved_model/`
* Converts SavedModel → TFLite (`.tflite`)

## Real-Time Inference

Run:

```bash
python scripts/detect.py
```

* Opens webcam, runs TFLite inference
* Displays bounding boxes and labels in real-time

## MLOps Pipeline

Reproduce the full pipeline with:

```bash
dvc repro
```

Stages: `prepare_data` → `train_model` → `evaluate_model` → `export_model`

## Experiment Tracking

* Launch MLflow UI:

  ```bash
  mlflow ui
  ```
* Compare runs, hyperparameters, and metrics

## Versioning and Reproduction

* Commit code changes with Git
* Version data and models with DVC:

  ```bash
  dvc add data/raw/
  dvc add models/yolov4-tiny/last.weights
  ```
* Use `dvc metrics diff` to compare metrics between commits

## Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a pull request

## License

This project is released under the [MIT License](LICENSE).
