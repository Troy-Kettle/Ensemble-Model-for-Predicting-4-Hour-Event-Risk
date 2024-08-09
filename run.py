

# Load and preprocess test data
def load_and_preprocess_test_data(file_path, scaler, selector, poly):
    data = pd.read_excel(file_path)
    
    # Advanced imputation
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Feature engineering
    data_imputed['BP_RATIO'] = data_imputed['SYSTOLIC_BP'] / data_imputed['HEART_RATE']
    data_imputed['TEMP_HR_RATIO'] = data_imputed['TEMPERATURE'] / data_imputed['HEART_RATE']
    data_imputed['O2_HEART_PRODUCT'] = data_imputed['O2_SATS'] * data_imputed['HEART_RATE']
    
    X = data_imputed.drop('4_HOURS_FROM_ANNOTATED_EVENT', axis=1)
    y = data_imputed['4_HOURS_FROM_ANNOTATED_EVENT']
    
    # Polynomial features
    X_poly = poly.transform(X)
    
    # Feature selection
    X_selected = selector.transform(X_poly)
    
    # Scaling
    X_scaled = scaler.transform(X_selected)
    
    return X_scaled, y

# Load models and preprocessors
def load_models_and_preprocessors(model_path, poly_path, scaler_path, selector_path):
    model = load_model(model_path)
    poly = joblib.load(poly_path)
    scaler = joblib.load(scaler_path)
    selector = joblib.load(selector_path)
    xgb_model = joblib.load('xgb_model.pkl')
    
    return model, poly, scaler, selector, xgb_model

# Evaluate model and plot results
def evaluate_model(model, X_test, y_test, xgb_model):
    # Predict with the ensemble
    def get_ensemble_prediction(nn_model, xgb_model, X):
        nn_pred = nn_model.predict(X)
        xgb_pred = xgb_model.predict_proba(X)[:, 1]
        return 0.6 * nn_pred.flatten() + 0.4 * xgb_pred
    
    y_pred_ensemble = get_ensemble_prediction(model, xgb_model, X_test)
    y_pred_class = (y_pred_ensemble > 0.5).astype(int)
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_class))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_class))
    
    print("\nROC AUC Score:")
    print(roc_auc_score(y_test, y_pred_ensemble))
    
    # Plot performance
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred_class), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')

    plt.subplot(1, 2, 2)
    plt.hist(y_pred_ensemble, bins=20, edgecolor='k', alpha=0.7)
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    model_path = 'neural_network_model.h5'
    poly_path = 'poly_preprocessor.pkl'
    scaler_path = 'scaler.pkl'
    selector_path = 'selector.pkl'
    test_file_path = 'ANFIS_test.xlsx'
    
    # Load models and preprocessors
    model, poly, scaler, selector, xgb_model = load_models_and_preprocessors(
        model_path, poly_path, scaler_path, selector_path
    )
    
    # Load and preprocess test data
    X_test, y_test = load_and_preprocess_test_data(test_file_path, scaler, selector, poly)
    
    # Evaluate and plot results
    evaluate_model(model, X_test, y_test, xgb_model)

if __name__ == '__main__':
    main()
