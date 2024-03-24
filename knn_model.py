from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNModel:
    def __init__(self):
        self.results = []

    def test_configurations(self, X_train, X_test, y_train, y_test, k_range):
        for k in k_range:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)  # Fit model on training data
            y_pred = model.predict(X_test)  # Predict on test data
            accuracy = accuracy_score(y_test, y_pred)  # Evaluate accuracy on test data
            self.results.append({'K': k, 'Accuracy': accuracy})
    
    def get_best_configuration(self):
        best_result = max(self.results, key=lambda x: x['Accuracy'])
        return best_result
    
    def run(self, X_train, X_test, y_train, y_test, k_range=range(1, 31)):
        self.test_configurations(X_train, X_test, y_train, y_test, k_range)
        return self.get_best_configuration()
