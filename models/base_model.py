import joblib
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BaseModel:
    def __init__(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any) -> None:
        """
        Initialize the base model with training and testing data.

        :param X_train: Training data features
        :param X_test: Testing data features
        :param y_train: Training data labels
        :param y_test: Testing data labels
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.results: List[Dict[str, Union[int, float]]] = []

    def fit(self) -> None:
        """
        Fit the model on the training data. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method!")

    def predict(self, X: Any) -> Any:
        """
        Predict the labels for the given data.

        :param X: Data to predict labels for
        :return: Predicted labels
        """
        if not self.model:
            raise Exception("Model not trained!")
        return self.model.predict(X)

    def evaluate(self) -> Dict[str, Union[float, Any]]:
        """
        Evaluate the model on the testing data.

        :return: A dictionary containing evaluation metrics
        """
        y_pred = self.predict(self.X_test)
        metrics = {
            'Accuracy': round(accuracy_score(self.y_test, y_pred), 3),
            'Precision': round(precision_score(self.y_test, y_pred, average='weighted'), 3),
            'Recall': round(recall_score(self.y_test, y_pred, average='weighted'), 3),
            'F1 Score': round(f1_score(self.y_test, y_pred, average='weighted'), 3),
            # 'Confusion Matrix': confusion_matrix(self.y_test, y_pred)
        }
        return metrics

    def save_model(self, file_path: str) -> None:
        """
        Save the trained model to a file.

        :param file_path: The path to save the model file
        """
        if not self.model:
            raise Exception("No model to save!")
        joblib.dump(self.model, file_path)

    def load_model(self, file_path: str) -> None:
        """
        Load a model from a file.

        :param file_path: The path to the model file
        """
        self.model = joblib.load(file_path)

    def get_best_configuration(self) -> Optional[Dict[str, Union[int, float]]]:
        """
        Get the best configuration based on the stored results.

        :return: A dictionary containing the best configuration parameters and their corresponding accuracy
        """
        return max(self.results, key=lambda x: x['Accuracy']) if self.results else None

    def run(self, **kwargs) -> Optional[Dict[str, Union[int, float]]]:
        """
        Run the model training and return the best configuration. Subclasses should implement this method.

        :return: The best configuration parameters and their corresponding accuracy
        """
        self.fit()
        return self.get_best_configuration()

    def test_configurations(self) -> None:
        """
        Test different configurations of the model. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method!")
