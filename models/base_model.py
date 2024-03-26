import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.results = []

    def fit(self):
        raise NotImplementedError("Subclasses should implement this method!")

    def predict(self, X):
        if self.model:
            return self.model.predict(X)
        raise Exception("Model not trained!")

    def evaluate(self):
        y_pred = self.predict(self.X_test)
        metrics = {
            'Accuracy': round(accuracy_score(self.y_test, y_pred), 3),
            'Precision': round(precision_score(self.y_test, y_pred, average='weighted'), 3),
            'Recall': round(recall_score(self.y_test, y_pred, average='weighted'), 3),
            'F1 Score': round(f1_score(self.y_test, y_pred, average='weighted'), 3),
            'Confusion Matrix': confusion_matrix(self.y_test, y_pred)
        }
        return metrics

    def save_model(self, file_path):
        if self.model:
            joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)

    def get_best_configuration(self):
        if self.results:
            return max(self.results, key=lambda x: x['Accuracy'])
        return None

    def run(self, **kwargs):
        self.fit()
        return self.get_best_configuration()

    def test_configurations(self):
        raise NotImplementedError("Subclasses should implement this method!")
