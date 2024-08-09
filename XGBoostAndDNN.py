import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from xgboost import XGBClassifier
from optuna import create_study
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    
    # Advanced imputation
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    X = data_imputed.drop('4_HOURS_FROM_ANNOTATED_EVENT', axis=1)
    y = data_imputed['4_HOURS_FROM_ANNOTATED_EVENT']

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Feature selection using XGBoost
    xgb_selector = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = SelectFromModel(xgb_selector, prefit=False)
    X_selected = selector.fit_transform(X_poly, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    return X_scaled, y.values, scaler, selector, poly

def create_nn_model(input_shape, dropout_rate=0.5, learning_rate=0.001):
    input_layer = Input(shape=(input_shape,))
    
    x1 = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(input_layer)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout_rate)(x1)
    x1 = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout_rate)(x1)
    
    x2 = Dense(64, activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(input_layer)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout_rate)(x2)
    x2 = Dense(32, activation='tanh', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout_rate)(x2)
    
    merged = concatenate([x1, x2])
    
    output = Dense(1, activation='sigmoid')(merged)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

def objective(trial, X, y):
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    model = create_nn_model(X.shape[1], dropout_rate, learning_rate)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')
    
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr, pruning_callback],
        verbose=0
    )
    
    return max(history.history['val_accuracy'])

def train_nn_model(X, y):
    study = create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20)
    
    best_params = study.best_params
    best_model = create_nn_model(X.shape[1], best_params['dropout_rate'], best_params['learning_rate'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    
    history = best_model.fit(
        X, y,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    return best_model, history

def xgb_objective(trial, X, y, param_grid):
    params = {
        'max_depth': trial.suggest_categorical('max_depth', param_grid['max_depth']),
        'learning_rate': trial.suggest_categorical('learning_rate', param_grid['learning_rate']),
        'n_estimators': trial.suggest_categorical('n_estimators', param_grid['n_estimators']),
        'min_child_weight': trial.suggest_categorical('min_child_weight', param_grid['min_child_weight']),
        'subsample': trial.suggest_categorical('subsample', param_grid['subsample'])
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
    
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in skf.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb.predict(X_val)
        scores.append(accuracy_score(y_val, y_pred))
    
    return np.mean(scores)

def train_xgboost(X, y):
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0]
    }
    
    study = create_study(direction='maximize')
    study.optimize(lambda trial: xgb_objective(trial, X, y, param_grid), n_trials=20)
    
    best_params = study.best_params
    best_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
    best_xgb.fit(X, y)
    
    return best_xgb

def get_ensemble_prediction(nn_model, xgb_model, X):
    nn_pred = nn_model.predict(X)
    xgb_pred = xgb_model.predict_proba(X)[:, 1]
    return 0.6 * nn_pred.flatten() + 0.4 * xgb_pred  # Weighted average

def ts_controller_from_ensemble(membership_value):
    if membership_value < 0.2:
        return 0  # Very low chance
    elif 0.2 <= membership_value < 0.4:
        return 25  # Low chance
    elif 0.4 <= membership_value < 0.6:
        return 50  # Moderate chance
    elif 0.6 <= membership_value < 0.8:
        return 75  # High chance
    else:
        return 100  # Very high chance

def plot_performance(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'ANFIS.xlsx'

    X, y, scaler, selector, poly = load_and_preprocess_data(file_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Neural Network...")
    nn_model, history = train_nn_model(X_train, y_train)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    print("\nEvaluating Ensemble Model...")
    y_pred_ensemble = get_ensemble_prediction(nn_model, xgb_model, X_test)
    y_pred_class = (y_pred_ensemble > 0.5).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_class))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_ensemble))

    plot_performance(history)

    # Example user input
    user_input = [180, 120, 20, 95, 39.5]
    user_input_poly = poly.transform([user_input])
    user_input_selected = selector.transform(user_input_poly)
    user_input_scaled = scaler.transform(user_input_selected)

    membership_value = get_ensemble_prediction(nn_model, xgb_model, user_input_scaled)[0]
    print(f"\nPredicted likelihood of event happening: {membership_value:.2f}")

    event_chance = ts_controller_from_ensemble(membership_value)
    print(f"Chance of event happening: {event_chance}%")

if __name__ == '__main__':
    main()
