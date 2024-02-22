# MNIST Classification with Convolutional Neural Network (CNN)

This repository contains a Jupyter notebook demonstrating the implementation of a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. The code is written in Python using the PyTorch library. The notebook consists of the following sections:

## Overview

The notebook begins by importing necessary libraries, setting the device to GPU if available, and loading the MNIST dataset. It uses the torchvision library for dataset handling and transformations.

## Data Visualization

The notebook includes code to visualize a few sample images from the MNIST dataset using Matplotlib.

## Convolutional Neural Network (CNN) Architecture

A simple CNN architecture is defined using the PyTorch `nn.Module` class. The architecture includes convolutional layers, ReLU activation, and max-pooling layers. The model is defined to take grayscale images of size 28x28 and outputs class probabilities for digits 0 to 9.

## Model Training

The CNN model is trained using the training set with stochastic gradient descent (SGD) as the optimizer and cross-entropy loss as the criterion. The training loop is executed for a specified number of epochs, and training and testing losses, along with test accuracy, are printed for each epoch.

## Results Visualization

The notebook includes code to visualize the training and testing loss curves over epochs.

## Inference and Visualization

The trained model is used to make predictions on a sample test image. The model's output logits are converted to class probabilities using softmax, and the results are visualized with Matplotlib.

## Requirements

- Python 3.x
- PyTorch
- Matplotlib
- torchvision

## How to Use

1. Ensure that the required dependencies are installed.
2. Run the notebook in a Jupyter environment.
3. Follow the sections step by step to understand the MNIST classification using a CNN.

Feel free to explore and modify the code according to your requirements.
