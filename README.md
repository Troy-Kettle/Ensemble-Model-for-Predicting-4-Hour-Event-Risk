# ANFIS Model: Predicting 4-Hour Event Risk

This project aims to develop an ensemble model to predict the likelihood of an event happening within 4 hours using a combination of Neural Networks and XGBoost classifiers. The model takes a variety of patient vitals and features to produce a probability score which is then used to determine the risk level.

## Table of Contents
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Ensemble Model Evaluation](#ensemble-model-evaluation)
- [User Input Prediction](#user-input-prediction)
- [Visualization](#visualization)
- [Contact](#contact)

## Installation
To run this project, you need to have Python installed along with several libraries. You can install the required libraries using the following command:

```bash
pip install pandas numpy scikit-learn tensorflow imbalanced-learn xgboost optuna matplotlib seaborn
```

## Project Structure
- `anfis_model.py`: Main script containing all the functions and the entry point (`main()` function) to run the project.
- `ANFIS.xlsx`: Excel file containing the dataset.

## Usage
### Load and Preprocess Data:
1. Load data from `ANFIS.xlsx`.
2. Perform advanced imputation using KNN.
3. Engineer new features.
4. Apply polynomial feature expansion.
5. Perform feature selection using XGBoost.
6. Standardize the selected features.

### Train Neural Network:
1. Define and compile a neural network model with hyperparameter tuning using Optuna.
2. Train the model with early stopping and learning rate reduction.

### Train XGBoost:
1. Define and train an XGBoost model with hyperparameter tuning using Optuna.
2. Perform cross-validation to select the best hyperparameters.

### Evaluate Ensemble Model:
1. Combine predictions from both models using a weighted average.
2. Evaluate performance using confusion matrix, classification report, and ROC AUC score.

### User Input Prediction:
1. Transform user input data using the same preprocessing pipeline.
2. Get ensemble prediction and determine the risk level.

### Visualization:
- Plot the training history of the neural network model to visualize accuracy and loss over epochs.

## Data Preprocessing
The data is preprocessed using the following steps:
1. Imputation: Missing values are imputed using the KNN algorithm.
2. Feature Engineering: New features are created from the existing ones to capture more information.
3. Polynomial Features: Polynomial features of degree 2 are generated.
4. Feature Selection: XGBoost is used to select the most important features.
5. Standardization: The selected features are standardized to have zero mean and unit variance.

## Model Training
### Neural Network
A neural network model is trained with the following steps:
1. Architecture: The model includes two parallel branches with different activation functions and dropout layers.
2. Compilation: The model is compiled using Adam optimizer with a binary cross-entropy loss function.
3. Training: The model is trained with early stopping and learning rate reduction callbacks. Hyperparameters are tuned using Optuna.

### XGBoost
An XGBoost model is trained with the following steps:
1. Parameter Tuning: Hyperparameters are tuned using Optuna.
2. Cross-Validation: The model is evaluated using stratified k-fold cross-validation.

## Ensemble Model Evaluation
The ensemble model combines predictions from both the neural network and XGBoost models. The performance is evaluated using:
- Confusion Matrix
- Classification Report
- ROC AUC Score

## User Input Prediction
To predict the likelihood of an event based on user input:
1. The input data is transformed using the same preprocessing pipeline.
2. The ensemble model produces a probability score.
3. A risk level is assigned based on the score.

## Visualization
The training history of the neural network model is plotted to visualize the accuracy and loss over epochs.

## Contact
For any queries or support, please contact Troy Kettle at troykettle19@gmail.com