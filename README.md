# DeepLearning-Experiments


# 1. Regression (California Housing Dataset)

## Problem:
The objective of this task is to predict **housing prices** using the **California Housing dataset**, a well-known dataset for regression tasks. The model is designed to estimate the **median house value** based on features like **median income**, **average rooms**, **latitude**, etc.

## Model:
A simple **Multilayer Perceptron (MLP)** architecture is used for the regression task. The model consists of the following layers:
- **Input Layer**: Accepts features like income, rooms, etc.
- **Hidden Layers**: Several Dense layers with **ReLU activation**, and regularization via **Dropout** and **BatchNormalization**.
- **Output Layer**: A single neuron for regression (predicting the target variable).

## Artifacts:
The following artifacts are provided for analysis:
- **Training History Plots**: Visualizations of loss and accuracy over epochs.
- **Model Visualization**: A diagram of the neural network architecture.
- **Regression Metrics**: MSE, RMSE, MAE, and additional metrics to evaluate model performance.
- **Error Analysis**: Visualizations to inspect model residuals and prediction errors.
