from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, X_train, X_test, y_train, y_test, n_estimators_list=[100], max_depth_list=[None], random_state=42):
        super().__init__(X_train, X_test, y_train, y_test)
        self.n_estimators_list = n_estimators_list
        self.max_depth_list = max_depth_list
        self.random_state = random_state

    def fit(self):
        param_grid = {
            'n_estimators': self.n_estimators_list,
            'max_depth': self.max_depth_list,
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state), 
            param_grid, 
            cv=5, 
            scoring='accuracy', 
            verbose=1, 
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)

        self.model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Store both parameters and score in the results
        self.results.append({
            'params': best_params,
            'score': best_score
        })

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, average='weighted'),
            'Recall': recall_score(self.y_test, y_pred, average='weighted'),
            'F1 Score': f1_score(self.y_test, y_pred, average='weighted'),
            'Classification Report': classification_report(self.y_test, y_pred)
        }
        return metrics

    def run(self):
        self.fit()
        evaluation_results = self.evaluate()

        for metric, value in evaluation_results.items():
            if metric != 'Classification Report':
                print(f"{metric}: {round(value, 3)}")
            else:
                print(value)
                
        return self.results[-1]  # Consider returning a more comprehensive result
