import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming each model class (KNNModel, etc.) is in the models package
from models.knn_model import KNNModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('Target', axis=1)
    y = data['Target']
    X_scaled = preprocess_data(X)
    return X_scaled, y

def run_model(X, y, model_class, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)

    model = model_class(X_train, X_test, y_train, y_test)
    best_config = model.run(**kwargs)

    # Evaluate the best model
    evaluation_results = model.evaluate()
    print("Best KNN Configuration: K =", best_config["K"])
    print("Evaluation Results for the Best Model:")
    for metric, value in evaluation_results.items():
        print(f"{metric}\n{value}\n")

    return best_config, model.model

datasets_path = 'datasets/'
for filename in os.listdir(datasets_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(datasets_path, filename)
        print(f"Processing {filename}...")
        print()

        X, y = load_and_preprocess_data(filepath)

        models = []

        # Run KNN Model
        print("Running KNN Model...")
        best_knn_config, knn_model = run_model(X, y, KNNModel, k_range=range(1, 31))
        models.append(('knn', knn_model))

        # Uncomment to run other models with their respective configurations
        # print("Running Logistic Regression Model...")
        # best_lr_config, lr_model = run_model(X, y, LogisticRegressionModel, param_range=some_range)
        # print("Best Logistic Regression Configuration:", best_lr_config)
        # models.append(('lr', lr_model))

        # print("Running SVM Model...")
        # best_svm_config, svm_model = run_model(X, y, SVMModel, param_range=some_range)
        # print("Best SVM Configuration:", best_svm_config)
        # models.append(('svm', svm_model))

        # print("Running Random Forest Model...")
        # best_rf_config, rf_model = run_model(X, y, RandomForestModel, param_range=some_range)
        # print("Best Random Forest Configuration:", best_rf_config)
        # models.append(('rf', rf_model))

        # TODO add ensemble model based on  models [] 

        print("=====================================")
