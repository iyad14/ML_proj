from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from math import sqrt
from .base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test):
        super().__init__(X_train, X_test, y_train, y_test)

    def fit(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(self.X_train, self.y_train)

    def test_configurations(self):
        n = len(self.X_train)
        sqrt_n = int(sqrt(n))
        k_range = range(1, sqrt_n + 1)  # Update k_range to be from 1 to sqrt(n)
        
        for k in k_range:
            self.fit(n_neighbors=k)
            y_pred = self.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results.append({'K': k, 'Accuracy': accuracy})
        
        best_config = self.get_best_configuration()
        if best_config:
            self.fit(n_neighbors=best_config['K'])  # Refit model with best K value

    def run(self, **kwargs):
        self.test_configurations()
        return self.get_best_configuration()
