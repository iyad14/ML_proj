from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class RandomForestModel(BaseEstimator):
    def __init__(self, categorical_cols, numerical_cols, n_estimators_list=[100], max_depth_list=[None], random_state=42):
        self.categorical_cols = categorical_cols
        self.numerical_cols = numerical_cols
        self.n_estimators_list = n_estimators_list
        self.max_depth_list = max_depth_list
        self.random_state = random_state

    def _preprocessor(self):
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)])
        return preprocessor

    def fit(self, X, y):
        preprocessor = self._preprocessor()
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(random_state=self.random_state))])
        param_grid = {
            'classifier__n_estimators': self.n_estimators_list,
            'classifier__max_depth': self.max_depth_list,
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)
        self.best_params_ = grid_search.best_params_
        self.best_estimator_ = grid_search.best_estimator_

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_with_info(self, X):
        predictions = self.predict(X)
        probabilities = self.best_estimator_.predict_proba(X)
        max_probabilities = np.max(probabilities, axis=1)
        classes = self.best_estimator_.classes_
        prediction_info = []
        for i, prediction in enumerate(predictions):
            info = {
                'Prediction': prediction,
                'Certainty Score': max_probabilities[i],
                'All Class Probabilities': dict(zip(classes, probabilities[i]))
            }
            prediction_info.append(info)
        return prediction_info

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print("Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y, y_pred))
        return accuracy
