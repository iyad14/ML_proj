import os
import logging
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import KNNModel, LogisticRegressionModel # Add others here

from utils import load_and_preprocess_data, create_ensemble_model
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    datasets_path = 'datasets/'
    for filename in os.listdir(datasets_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(datasets_path, filename)
            logging.info(f"Processing {filename}")

            X, y = load_and_preprocess_data(filepath)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            models_info = [
                ('KNN', KNNModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                ('Logistic Regression', LogisticRegressionModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                # Add other models as needed
            ]
            
            models = run_models(models_info)

            if len(models) > 1:
                ensemble_model = create_ensemble_model(models)
                # Fit, evaluate, and use ensemble_model as needed

            logging.info("=" * 50)

def run_models(models_info):
    models = []
    for model_name, model_class, model_params in models_info:
        logging.info(f"Running {model_name} Model...")
        
        # Instantiate the model with the provided parameters
        model = model_class(**model_params)
        best_config = model.run()  # Run the model to find the best configuration
        evaluation_results = model.evaluate()  # Evaluate the model to get performance metrics
        
        # Print the best configuration and evaluation results
        logging.info(f"Evaluation Results for {model_name}:")
        if best_config:
            logging.info(f"Best Configuration: {best_config}")
        for metric, value in evaluation_results.items():
            logging.info(f"{metric}: {value}")

        models.append((model_name, model))

    return models

if __name__ == "__main__":
    main()