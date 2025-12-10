# Plant Disease Detection

A simple project demonstrating plant disease classification using image-based deep learning. The repository contains a Jupyter Notebook (`Plant_Disease_Detection.ipynb`) that covers dataset handling, model training and evaluation.

---

## Overview

This project detects plant leaf diseases using a convolutional neural network. Users can load a dataset, preprocess images, train a model, and evaluate accuracy through the notebook.

---

## Features

* Image preprocessing and augmentation
* CNN/Transfer Learning-based classification
* Training, validation, and evaluation workflow
* Basic visualizations (accuracy/loss curves, example predictions)

---

## Requirements

Create a virtual environment and install dependencies such as:

```
numpy
pandas
matplotlib
opencv-python
tensorflow
scikit-learn
jupyter
```

Install via:

```
pip install -r requirements.txt
```

(or install packages manually if no requirements file exists.)

---

## Usage

### Run Notebook

```
jupyter notebook Plant_Disease_Detection.ipynb
```

The notebook guides you through:

* Loading dataset
* Preprocessing
* Model training
* Evaluation

---

## Dataset

Use any plant leaf image dataset organized by class folders. Popular choice: PlantVillage dataset (public). Update notebook paths accordingly.

---

## Model Summary

* Input preprocessing (resize + normalization)
* CNN or transfer learning (e.g., EfficientNet/MobileNet)
* Softmax output for multi-class classification
* Metrics: accuracy, precision, recall, confusion matrix

---

## Results

The notebook typically includes:

* Accuracy/loss curves
* Confusion matrix
* Sample predictions

---
