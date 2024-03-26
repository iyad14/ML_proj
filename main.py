import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

# Import your model classes
from models.knn_model import KNNModel
# from models.logistic_regression_model import LogisticRegressionModel
# from models.random_forest_model import RandomForestModel
# from models.svm_model import SVMModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    model = model_class(X_train, X_test, y_train, y_test)
    best_config = model.run(**kwargs)
    evaluation_results = model.evaluate()

    # Logging the results
    logging.info(f"Best Configuration: {best_config}")
    logging.info("Evaluation Results for the Best Model:")
    for metric, value in evaluation_results.items():
        logging.info(f"{metric}: {value}")

    return best_config, model.model

def create_ensemble_model(models):
    ensemble = VotingClassifier(estimators=models, voting='soft')
    return ensemble

datasets_path = 'datasets/'
for filename in os.listdir(datasets_path):
    if filename.endswith('.csv'):
        filepath = os.path.join(datasets_path, filename)
        logging.info(f"Processing {filename}")

        X, y = load_and_preprocess_data(filepath)
        models = []

        # Run KNN Model
        logging.info("Running KNN Model...")
        knn_config, knn_model = run_model(X, y, KNNModel)
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

        # Create and evaluate an ensemble model
        if len(models) > 1:
            ensemble_model = create_ensemble_model(models)
            # You can fit the ensemble model on the entire dataset or a separate training set
            # and evaluate it as needed

        logging.info("=====================================")
