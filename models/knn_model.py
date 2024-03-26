import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        """
        Initialize the KNN model with training and testing data.

        :param X_train: Training data features
        :param X_test: Testing data features
        :param y_train: Training data labels
        :param y_test: Testing data labels
        """
        super().__init__(X_train, X_test, y_train, y_test)

    def fit(self, n_neighbors: int = 5) -> None:
        """
        Fit the KNN model using the specified number of neighbors.

        :param n_neighbors: Number of neighbors to use
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self) -> None:
        """
        Test different configurations of the KNN model to find the best number of neighbors.
        """
        n = len(self.X_train)
        sqrt_n = int(np.sqrt(n))
        parameters = {'n_neighbors': range(1, sqrt_n + 1)}

        grid_search = GridSearchCV(KNeighborsClassifier(), parameters, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_parameters = grid_search.best_params_
        self.results.append({'K': best_parameters['n_neighbors'], 'Accuracy': grid_search.best_score_})

        # Refit the model with the best number of neighbors
        self.fit(n_neighbors=best_parameters['n_neighbors'])

    def run(self, **kwargs) -> dict:
        """
        Run the KNN model training and hyperparameter tuning process.

        :return: The best configuration parameters and their corresponding accuracy
        """
        self.test_configurations()
        return self.get_best_configuration()
