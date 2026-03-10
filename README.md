# Simple Neural Networks from Scratch (Classification & Regression)

This repository contains a simple, educational implementation of a Multilayer Perceptron (MLP) built entirely from scratch using Python and numpy. The project is designed to demonstrate the fundamental inner workings of neural networks, including forward propagation, backpropagation, and Stochastic Gradient Descent (SGD).

To show how the network architecture adapts to different types of problems, the repository includes two distinct variations of the model:
- Classification Network: Designed to categorize data into discrete classes (e.g., recognizing handwritten digits).
- Regression Network: Modified to predict continuous numerical values (e.g., approximating mathematical functions or predicting prices).

# 🚀 Features
Built from Scratch: Core math and logic are implemented using only numpy, without relying on heavy deep learning frameworks like TensorFlow or PyTorch.

Stochastic Gradient Descent (SGD): Includes mini-batch training for efficient weight and bias updates.

Classification Model: Uses Sigmoid activations in hidden layers and evaluates performance based on accuracy (correct predictions vs. total).

Regression Model: Features a linear output layer (no activation function on the final layer) and evaluates performance using Mean Squared Error (MSE).

Data Logging: Hooks for exporting training progress (e.g., epoch accuracy/loss) to CSV files for plotting and analysis.

# 📂 Project Structure
- classification_network.py - The core network class tailored for classification tasks.
- regression_network.py - The modified network class adapted for continuous value prediction.
- data_loader.py - Utility scripts for generating synthetic regression data or loading datasets like MNIST.

# 🛠️ Prerequisites
To run this project, you need Python 3 installed along with the following library:

- pandas
