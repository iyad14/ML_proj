import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Initialize the Logistic Regression model with training and testing data.
        """
        super().__init__(X_train, X_test, y_train, y_test)

    def fit(self, **kwargs) -> None:
        """
        Fit the Logistic Regression model using specified hyperparameters.
        """
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**kwargs, max_iter=10000))
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self) -> None:
        self.results = []
        # Create a pipeline with scaling and logistic regression
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))

        # Define the parameter grid
        param_grid = {
            'logisticregression__penalty': ['l2'],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__solver': ['lbfgs'],  # Simplified for compatibility with 'l2'
        }

        # Initialize GridSearchCV with the pipeline as the estimator
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

        # Fit GridSearchCV to find the best configuration
        grid_search.fit(self.X_train, self.y_train)

        # Extract the best model, parameters, and the corresponding score
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Update the model attribute with the best found model
        self.model = best_model

        # Append the results with the best configuration and its score
        self.results.append({
            'penalty': best_params['logisticregression__penalty'],
            'C': best_params['logisticregression__C'],
            'solver': best_params['logisticregression__solver'],
            'Accuracy': best_score
        })

        
    def run(self):
        """
        Run the logistic regression training and hyperparameter tuning process.

        :return: The best configuration parameters and their corresponding accuracy
        """
        self.test_configurations()
        return self.get_best_configuration()
