# ML & From-Scratch Neural Networks

A comprehensive exploration of Machine Learning and Deep Learning architectures applied to regression and classification tasks.

## Project Overview

The repository is structured into two primary sub-problems, each addressed using both traditional ML algorithms and custom Neural Networks.

### 1. California Housing Price Prediction (Regression)
An analytical approach to predicting real estate prices in California. 
* **Methodology:** Extensive feature engineering (e.g., population density, rooms-per-household ratios) followed by model optimization.
* **Algorithms:** Random Forest, Gradient Boosting Regressor (GBR), K-Nearest Neighbors (KNN), and Support Vector Regression (SVR).
* **Hyperparameter Tuning:** Systematic analysis of parameters such as `n_estimators`, `max_depth`, and `learning_rate` to optimize the bias-variance tradeoff.
* **Deep Learning:** A custom regression neural network implemented to capture non-linear relationships in tabular housing data.

### 2. MNIST Digit Classification (Classification)
The classic "Hello World" of computer vision, solved through multiple lenses.
* **Traditional ML:** Comparison of various classification algorithms on high-dimensional pixel data.
* **Custom Neural Network:** A multi-layer perceptron (MLP) built strictly from scratch using NumPy. This implementation includes:
    * Forward and backpropagation logic.
    * Mini-Batch Stochastic Gradient Descent (SGD).
    * Custom weight initialization and activation functions.

---

## Repository Structure & Key Notebooks

### Regression (California Housing)
* **Machine Learning Analysis:** [`ML_Regression/ML_Regression.ipynb`](./ML_Regression/ML_Regression.ipynb)  
    *Detailed GridSearch analysis and comparison of tree-based vs. kernel-based models.*
* **Neural Network Implementation:** [`Regresyjny/housing_network.ipynb`](./Regresyjny/housing_network.ipynb)  
    *The custom-built architecture for continuous value prediction.*

### Classification (MNIST)
* **Machine Learning Analysis:** [`ML_Classification/Ml_Classification.ipynb`](./ML_Classification/Ml_Classification.ipynb)  
    *Classification benchmarks and pixel-data scaling experiments.*
* **Neural Network Implementation:** [`Klasyfikacyjny/classification_network.ipynb`](./Klasyfikacyjny/classification_network.ipynb)  
    *Step-by-step demonstration of the from-scratch network performance.*

### Results & Summary
* **Final Comparison & Executive Summary:** [Placeholder for summary_results.ipynb]  
    *A consolidated view of performance metrics across all models.*

---

## Technical Details: Neural Networks from Scratch

A core highlight of this project is the **avoidance of black-box libraries** (TensorFlow, PyTorch, Keras) for the deep learning component. 

* **Backpropagation & Matrix Calculus:** Direct implementation of partial derivatives to compute error gradients layer-by-layer.
* **Custom Optimization:** Manual implementation of weight and bias updates via Mini-Batch SGD.
* **Alternative Cost Function:** Replacement of the standard quadratic cost with the **Cross-Entropy cost function** to mathematically prevent learning stagnation.
* **Regularization & Initialization:** Implementation of **L2 Regularization (Weight Decay)** to explicitly combat overfitting, alongside an optimized **Weight Initialization** strategy (scaling initial weights by the square root of input connections) to prevent vanishing/exploding gradients.

---

<p align="center">
  <b>Elements of Artificial Intelligence</b><br>
  AGH 2026
</p>
