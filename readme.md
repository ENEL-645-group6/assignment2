# Garbage Classification with Image and Text Features

## General Workflow
Get dataset → Run ENEL645_group6_a2_garbage_model_train.py to get best_model.pth → Run ENEL645_group6_a2_garbage_model_test.py to get the results

## Running Locally

The code is designed to automatically check for GPU availability and use it if available. Otherwise, it will use MPS (Apple Silicon) if available. Otherwise, it will use CPU.

1. Clone the repository
2. Download the datasets and place them in the project root
   - Dataset location on TALC: `/work/TALC/enel645_2025w/garbage_data`
   - Dataset size is about 15GB
3. Update the dataset path in `ENEL645_group6_a2_garbage_model_train.py`
4. Run `ENEL645_group6_a2_garbage_model_train.py` to get best_model.pth
5. Run `ENEL645_group6_a2_garbage_model_test.py` to get the results

### Expected Directory Structure
```
.
├── CVPR_2024_dataset_Test
├── CVPR_2024_dataset_Train
├── CVPR_2024_dataset_Val
├── ENEL645_group6_a2_garbage_model_test.py
├── ENEL645_group6_a2_garbage_model_train.py
├── __pycache__
└── garbage_ML_test.py
```

## Running on TALC
1. Copy code from `ENEL645_group6_a2_garbage_model_train.py` and `ENEL645_group6_a2_garbage_model_test.py` to TALC
2. Update the dataset path in `ENEL645_group6_a2_garbage_model_train.py`
   - Dataset location on TALC: `/work/TALC/enel645_2025w/garbage_data`
3. Create `.slurm` file to run the `ENEL645_group6_a2_garbage_model_train.py`
4. Run `ENEL645_group6_a2_garbage_model_train.py` to get best_model.pth
5. Create `.slurm` file to run the `ENEL645_group6_a2_garbage_model_test.py`
6. Run `ENEL645_group6_a2_garbage_model_test.py` to get the results

A multimodal deep learning model that classifies garbage images using both visual features and textual information from filenames.

## Prerequisites

- Python 3.x
- PyTorch
- Transformers
- torchvision

Install dependencies:

```bash
pip install torch torchvision transformers
```

## Dataset Structure

Download the following datasets and place them in the project root:
- `CVPR_2024_dataset_Train/`
- `CVPR_2024_dataset_Val/`
- `CVPR_2024_dataset_Test/`

Your working directory should look like this:
```
.
├── CVPR_2024_dataset_Test
├── CVPR_2024_dataset_Train
├── CVPR_2024_dataset_Val
├── ENEL645_group6_a2_garbage_model_test.py
├── ENEL645_group6_a2_garbage_model_train.py
├── __pycache__
├── best_model.pth
└── garbage_ML_test.py
```

## Usage

1. Train the model:

```bash
python ENEL645_group6_a2_garbage_model_train.py
```

2. Test the model:

```bash
python ENEL645_group6_a2_garbage_model_test.py
```

## Model Details

- Image Features: MobileNetV2
- Text Features: BERT
- Combined with custom fusion layers
- Supports CUDA (GPU), MPS (Apple Silicon), and CPU

## Note

Large dataset files and model weights are not included in the repository. Please obtain them from the provided remote server.

