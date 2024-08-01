Ensemble Model for Predicting 4-Hour Event Risk
This project focuses on developing an ensemble model to predict the likelihood of an event occurring within 4 hours using a combination of Neural Networks and XGBoost classifiers. The model integrates various patient vitals and features to produce a probability score, which is used to determine the risk level.

Table of Contents
Installation
Project Structure
Usage
Data Preprocessing
Model Training
Ensemble Model Evaluation
User Input Prediction
Visualization
Contact
Installation
Ensure you have Python installed along with the required libraries. Install the necessary libraries using the following command:

bash
Copy code
pip install pandas numpy scikit-learn tensorflow imbalanced-learn xgboost optuna matplotlib seaborn
Project Structure
ensemble_model.py: Main script containing functions and the main() function to execute the project.
ANFIS.xlsx: Excel file containing the dataset.
Usage
Load and Preprocess Data:
Load Data: Import data from ANFIS.xlsx.
Imputation: Handle missing values using KNN imputation.
Feature Engineering: Create new features to enhance the model's information capacity.
Polynomial Features: Generate polynomial features of degree 2.
Feature Selection: Use XGBoost to select significant features.
Standardization: Standardize the features to have zero mean and unit variance.
Train Neural Network:
Model Architecture: Define a neural network with two parallel branches, dropout layers, and batch normalization.
Compilation: Compile the model with the Adam optimizer and binary cross-entropy loss.
Training: Train the model with early stopping and learning rate reduction, and optimize hyperparameters using Optuna.
Train XGBoost:
Parameter Tuning: Optimize hyperparameters using Optuna.
Cross-Validation: Evaluate model performance with stratified k-fold cross-validation.
Evaluate Ensemble Model:
Combine Predictions: Merge predictions from the neural network and XGBoost using a weighted average.
Performance Metrics: Assess model performance using confusion matrix, classification report, and ROC AUC score.
User Input Prediction:
Preprocess Input: Transform user input data using the same preprocessing pipeline.
Predict: Compute the ensemble prediction and assign a risk level based on the score.
Visualization:
Plot the training history of the neural network model to visualize accuracy and loss trends over epochs.
Data Preprocessing
The preprocessing steps include:

Imputation: Missing values are addressed using KNN imputation.
Feature Engineering: Additional features are created from existing data to capture more insights.
Polynomial Features: Polynomial feature expansion is applied to the data.
Feature Selection: XGBoost is used for feature selection.
Standardization: Features are standardized to have zero mean and unit variance.
Model Training
Neural Network
Architecture: The model features parallel branches with various activation functions and regularization.
Compilation: The model uses the Adam optimizer with binary cross-entropy loss.
Training: Includes early stopping and learning rate adjustment, with hyperparameters optimized using Optuna.
XGBoost
Hyperparameter Tuning: Parameters are tuned using Optuna.
Cross-Validation: The model is validated using stratified k-fold cross-validation to ensure robustness.
Ensemble Model Evaluation
The ensemble approach combines predictions from both neural network and XGBoost models. Performance is evaluated using:

Confusion Matrix
Classification Report
ROC AUC Score
User Input Prediction
For user-provided data:

Preprocessing: Apply the same transformations as used during training.
Prediction: Calculate the ensemble model's probability score.
Risk Level: Determine the risk level based on the predicted score.
Visualization
Training history of the neural network model is plotted to show accuracy and loss changes over epochs.

Contact
For inquiries or support, please reach out to Troy Kettle at troykettle19@gmail.com