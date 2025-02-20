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

# 2. Classification (Wildfire Prediction)

## Problem:
The objective of this task is to predict **wildfire occurrences** using sensor data. The model is designed to classify whether a wildfire will occur (**1 = Wildfire, 0 = No Wildfire**) based on real-time environmental conditions such as temperature, humidity, wind speed, and vegetation indices.

## Model:
A simple **Multilayer Perceptron (MLP)** architecture is used for the classification task. The model consists of the following layers:

- **Input Layer**: Accepts sensor-based environmental features.
- **Hidden Layers**: Several **Dense** layers with **ReLU activation**, with regularization using **Dropout** and **BatchNormalization**.
- **Output Layer**: A single neuron with **sigmoid activation** for binary classification.

## Artifacts:
The following artifacts are provided for analysis:

# 3. Image Classification (Fashion MNIST)

## Problem:
The objective of this task is to classify **fashion items** from the Fashion MNIST dataset. The model is designed to categorize images into one of **10 clothing classes** (e.g., T-shirt, Trouser, Dress, Sneaker) based on their grayscale images (28x28 pixels).

## Model:
A **Convolutional Neural Network (CNN)** architecture is used for the image classification task. The model consists of the following layers:

- **Input Layer:** Accepts 28x28 grayscale images.
- **Convolutional Layers:** Several **Conv2D** layers with **ReLU activation**, followed by **MaxPooling** for feature extraction.
- **Fully Connected Layers:** Flattened layer followed by **Dense** layers with **ReLU activation** for high-level feature learning.
- **Output Layer:** A **softmax** layer with 10 neurons for multi-class classification.

## Artifacts:
The following artifacts are provided for analysis:

- **Training History Plots:** Visualization of loss and accuracy over epochs.
- **Model Visualization:** A diagram of the CNN architecture.
- **Classification Metrics:** Accuracy, Precision, Recall, F1-score for **overall and per-class evaluation**.
- **Confusion Matrix:** A heatmap to analyze misclassifications per class.
- **ROC and PR Curves:** Visualization of model performance using **Receiver Operating Characteristic** and **Precision-Recall curves**.
- **Error Analysis:** Per-class analysis to inspect model predictions and identify **misclassifications** with sample images.

## Tracking & Monitoring:
- **Integration with Weights & Biases (wandb):** All training metrics, visualizations, and logs are recorded for experiment tracking.

- **Training History Plots**: Visualizations of loss and accuracy over epochs.
- **Model Visualization**: A diagram of the neural network architecture.
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score for overall and per-class evaluation.
- **ROC and PR Curves**: Visualization of model performance using Receiver Operating Characteristic and Precision-Recall curves.
- **Error Analysis**: Per-class analysis to inspect model predictions and identify misclassifications.
