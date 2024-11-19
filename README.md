# CIFAR-10 Image Classification from Scratch with PyTorch

## Description

This project implements a simple two-layer neural network from scratch using PyTorch to classify images from the CIFAR-10 dataset. It demonstrates the fundamental components of a neural network, including manual implementation of the forward pass, backward pass (backpropagation), activation functions (ReLU), and cross-entropy loss.

## Core Functionality

* Loads and preprocesses the CIFAR-10 dataset (ToTensor, Normalize, Flatten).
* Manually defines and implements a two-layer neural network.
* Implements ReLU activation function and its gradient.
* Implements cross-entropy loss function and its gradient.
* Conducts a training loop with gradient descent to update model weights.
* Evaluates the model on a validation set.

## Requirements

* Python 3.x
* PyTorch
* Torchvision

Install dependencies using pip:
```bash
pip install torch torchvision