# Sign Language Recognition using Hybrid GCN-CNN

A deep learning project for Indian Sign Language (ISL) recognition using a hybrid Graph Convolutional Network (GCN) and Convolutional Neural Network (CNN) architecture.

## Overview

This project combines graph neural networks with convolutional neural networks to recognize and classify Indian Sign Language gestures with high accuracy. The model processes hand keypoint coordinates and image data to make predictions.

## Dataset

This project uses the **Indian Sign Language (ISL) Dataset** from Kaggle:
- **Dataset Link:** https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
- **Description:** A comprehensive dataset containing images and keypoint coordinates for Indian Sign Language gestures
- **Usage:** Hand gesture recognition and sign language classification

## Project Structure

- `Sign_Language (1).ipynb` - Main Jupyter notebook with complete implementation
- `README.md` - Project documentation
- `.gitignore` - Git configuration files

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Manav250305/Sign-Language.git
cd Sign-Language
```

### 2. Download the Dataset
1. Visit [Kaggle ISL Dataset](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
2. Download the dataset
3. Extract the images to an `Indian/` directory in the project root
4. Extract or generate coordinate CSVs in the project root

### 3. Download Pre-trained Models (Optional)
If you want to use pre-trained models instead of training from scratch, download:
- `best_hybrid_gcn_cnn.pth` - Best performing hybrid model
- `optimized_gmtc_model.pth` - Optimized variant for faster inference

Place these `.pth` files in the project root directory.

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install torch torchvision torch-geometric scikit-image opencv-python mediapipe numpy pandas matplotlib jupyter
```

### 5. Run the Notebook
```bash
jupyter notebook "Sign_Language (1).ipynb"
```

## Key Features

- **Hybrid Architecture:** Combines GCN for spatial relationships and CNN for image features
- **Hand Keypoint Detection:** Processes MediaPipe keypoint coordinates
- **Multi-modal Learning:** Uses both image and coordinate data
- **Superpixel Segmentation:** SLIC algorithm for region-based features
- **PyTorch Implementation:** Built with PyTorch and PyTorch Geometric

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric
- scikit-image
- OpenCV
- MediaPipe
- NumPy, Pandas, Matplotlib
- Jupyter Notebook

## Model Files

The following model files are **not included** in the repository (use `.gitignore` to keep repo lightweight):
- `best_hybrid_gcn_cnn.pth` - Best performing hybrid model
- `optimized_gmtc_model.pth` - Optimized variant for faster inference

To use these models:
1. Train them using the notebook, or
2. Download from the original source and place in project root

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Dataset provided by [Kaggle - Indian Sign Language](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- Built with PyTorch, PyTorch Geometric, and scikit-image
