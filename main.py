import os
import logging
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from models import KNNModel, LogisticRegressionModel, SVMModel, RandomForestModel


from utils import load_and_preprocess_data, create_ensemble_model
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for different types of files
datasets_path = 'datasets/'
saved_models_path = os.path.join('models')
saved_models_path = os.path.join(saved_models_path, 'top_models')
results_path = os.path.join('data', 'results')
figures_path = os.path.join('data','figures')

def main():
    # Create subfolders if they don't exist
    for path in [saved_models_path, results_path, figures_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    if not os.path.exists(saved_models_path):
        os.makedirs(saved_models_path)

    for filename in os.listdir(datasets_path):
        if filename.endswith('.csv'):
            filepath = os.path.join(datasets_path, filename)
            logging.info(f"Processing {filename}")
            dataset_name = os.path.splitext(filename)[0]

            X, y = load_and_preprocess_data(filepath)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            models_info = [
                ('KNN', KNNModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                ('RandomForest', RandomForestModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                ('LogisticRegression', LogisticRegressionModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                ('SVM', SVMModel, {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}),
                # Ensemble model will be handled within run_models
            ]
            
            run_models(models_info, dataset_name, saved_models_path, results_path, X_train, X_test, y_train, y_test)

            logging.info("=" * 50)


def run_models(models_info, dataset_name, saved_models_path, results_path, X_train, X_test, y_train, y_test):
    models = []
    all_results = []

    for model_name, model_class, model_params in models_info:
        logging.info(f"Running {model_name} Model...")
        model = model_class(**model_params)
        model.run()
        evaluation_results = model.evaluate()

        evaluation_results['Model'] = model_name
        evaluation_results['Dataset'] = dataset_name
        all_results.append(evaluation_results)

        model_filename = os.path.join(saved_models_path, f"{model_name}_{dataset_name}.pkl")
        joblib.dump(model.model, model_filename)
        models.append((model_name, model.model))
        logging.info(f"Saved {model_name} model for {dataset_name} in {model_filename}")

    # If there are multiple models, create and evaluate the ensemble model
    if len(models) > 1:
        logging.info("Creating and Evaluating Ensemble Model...")
        ensemble_model = create_ensemble_model(models)
        ensemble_model.fit(X_train, y_train)

        # Evaluate the ensemble model
        predictions = ensemble_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        ensemble_results = {
            'Model': 'Ensemble',
            'Dataset': dataset_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
        all_results.append(ensemble_results)

        ensemble_model_filename = os.path.join(saved_models_path, f"EnsembleModel_{dataset_name}.pkl")
        joblib.dump(ensemble_model, ensemble_model_filename)
        logging.info(f"Saved ensemble model for {dataset_name} in {ensemble_model_filename}")

    # Save all results to a CSV file in the results subfolder
    results_filename = os.path.join(results_path, f"results_{dataset_name}.csv")
    pd.DataFrame(all_results).to_csv(results_filename, index=False)
    logging.info(f"Saved evaluation results for {dataset_name} in {results_filename}")

    return models

def visualize_performance(results_path, figures_path):
    # Find all results files
    results_files = [f for f in os.listdir(results_path) if f.startswith('results_') and f.endswith('.csv')]

    # Load and concatenate all results
    all_results = pd.concat([pd.read_csv(os.path.join(results_path, f)) for f in results_files])
    color_palette = ['red', 'blue', 'green']

    # Visualization of metrics for each model across all datasets
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        plt.title(f'{metric} of Models (Including Ensemble) Across Datasets')
        plt.ylabel(metric)
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.legend(title='Dataset')
        barplot = sns.barplot(data=all_results, x='Model', y=metric, hue='Dataset', palette=color_palette)
        # Add the text labels on top of each bar
        for p in barplot.patches:
            height = p.get_height()
            plt.text(p.get_x() + p.get_width() / 2., height + 0.01, f'{height:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f"{metric.lower()}_comparison.png"))
        plt.show()

if __name__ == "__main__":
    main()
    visualize_performance(results_path=results_path, figures_path=figures_path)