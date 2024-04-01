import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel  # Adjust import path based on your project structure

class KNNModel(BaseModel):
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
        super().__init__(X_train, X_test, y_train, y_test)

    def fit(self, n_neighbors: int = 5) -> None:
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self, param_grid: dict = None) -> None:
        if param_grid is None:
            n = len(self.X_train)
            sqrt_n = int(np.sqrt(n))
            param_grid = {'n_neighbors': range(1, sqrt_n + 1)}

        self.results = []  # Reset results for each test
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_
        self.results.append({'K': best_parameters['n_neighbors'], 'Accuracy': best_score})

        self.fit(n_neighbors=best_parameters['n_neighbors'])

    def run(self, **kwargs) -> dict:
        param_grid = kwargs.get('param_grid')
        self.test_configurations(param_grid=param_grid)
        return self.get_best_configuration()
