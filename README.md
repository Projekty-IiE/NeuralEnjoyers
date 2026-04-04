# 🧠 Machine Learning & Neural Networks: Classification and Regression

This repository contains an educational exploration of predictive algorithms, featuring both traditional Machine Learning models and Multilayer Perceptrons (MLPs) built entirely from scratch using Python and `numpy`. 

The project is designed to demonstrate the fundamental inner workings of these algorithms, including forward propagation, backpropagation, and hyperparameter tuning. To show how different architectures adapt to different types of problems, the repository is divided into **four separate reports**:

1. **Neural Network Classification:** An MLP built from scratch to recognize MNIST handwritten digits.
2. **Neural Network Regression:** A modified from-scratch MLP designed to estimate California Housing prices.
3. **Machine Learning Classification:** Implementation and tuning of standard ML algorithms on categorical data, evaluated on the same handwritten digits dataset.
4. **Machine Learning Regression:** Implementation and tuning of ensemble and distance-based ML algorithms (Random Forest, Gradient Boosting, KNN, SVR), benchmarked on the same California Housing dataset.
   
## 🚀 Features
* **Built from Scratch:** Core neural network math and logic (Stochastic Gradient Descent, Backpropagation) are implemented using only `numpy`, without relying on heavy frameworks like TensorFlow or PyTorch.
* **Hyperparameter Isolation & Analysis:** Systematically isolates individual model parameters (e.g., tree depth, penalty budgets, number of neighbors) to visualize their direct impact on model capacity and the bias-variance tradeoff.
* **Standard ML Pipelines:** Clean, structured implementations of industry-standard algorithms using `scikit-learn`, complete with Grid Search cross-validation.
* **Feature Engineering:** Demonstrates how to resolve real-world data biases through normalization and ratio creation.
* **Data Logging & Visualization:** Built-in hooks for exporting training progress, parameter testing, and accuracy metrics to CSV files for plotting and analysis.

## 📂 Project Structure
* `classification_network.ipynb` - The core Neural Network built from scratch tailored for classification tasks.
* `housing_network.ipynb` - The modified Neural Network from scratch adapted for continuous value prediction.
* `ML_Classification.ipynb` - Exploration and tuning of standard ML algorithms for classification.
* `ML_Regression.ipynb` - Exploration and tuning of standard ML algorithms for regression (California Housing).
* `data_loader.py` / `network.py` - Utility scripts for generating data, loading datasets (like MNIST), and core math functions.
* `*.csv` files - Logged experiment results for train/test performance across all models.

## 🛠️ Prerequisites
To run this project, you need Python 3 installed along with the following libraries:
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `folium` (for map visualizations)
