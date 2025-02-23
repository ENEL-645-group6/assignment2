# Garbage Classification with Image and Text Features

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

