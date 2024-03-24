from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

class LogisticRegressionModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def run(self):
        # Create a pipeline with scaling and logistic regression
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))

        # Hyperparameters to test
        param_grid = {
            'logisticregression__penalty': ['l2'],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
        }

        # Grid search with cross-validation
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        # Best model evaluation on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        return {
            'best_parameters': grid_search.best_params_,
            'test_accuracy': accuracy
        }
