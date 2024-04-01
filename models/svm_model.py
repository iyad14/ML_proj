from .base_model import BaseModel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class SVMModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)
        # Initialize with default parameters; can be updated later
        self.model = SVC()

    def fit(self):
        """
        Fit the model on the training data.
        """
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self):
        """
        Perform grid search to find the best SVM configuration.
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
        }
        grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5,scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        # Update the model with the best parameters found
        self.model = grid_search.best_estimator_

        # Store the best parameters and score
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        self.results.append({
            'C': self.best_params['C'],
            'kernel': self.best_params['kernel'],
            'Accuracy': self.best_score
        })

    def run(self):
        """
        Run grid search and then evaluate the model.

        :return: The best configuration parameters and their corresponding accuracy
        """
        self.test_configurations()
        return self.get_best_configuration()
