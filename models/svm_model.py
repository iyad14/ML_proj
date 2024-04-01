from .base_model import BaseModel  # Adjust the import path based on your project structure
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)

    def fit(self, C=1.0, kernel='rbf'):
        """
        Fit the SVM model using specified hyperparameters.
        """
        self.model = SVC(C=C, kernel=kernel)
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self):
        """
        Perform grid search to find the best SVM configuration.
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
        }
        grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        # Update the model with the best parameters found
        self.model = grid_search.best_estimator_

        # Store the best parameters and score in results
        self.results.append({
            'C': grid_search.best_params_['C'],
            'kernel': grid_search.best_params_['kernel'],
            'Accuracy': grid_search.best_score_
        })

    def run(self):
        """
        Run grid search and then evaluate the model.

        :return: The best configuration parameters and their corresponding accuracy
        """
        self.test_configurations()
        return self.get_best_configuration()
