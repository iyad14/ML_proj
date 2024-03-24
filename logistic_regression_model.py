from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogisticRegressionModel:
    def __init__(self):
        self.results = []

    def test_configurations(self, X_train, X_test, y_train, y_test, penalty_list=['l2'], C_values=[1.0]):
        for penalty in penalty_list:
            for C in C_values:
                model = LogisticRegression(penalty=penalty, C=C, max_iter=10000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.results.append({'Penalty': penalty, 'C': C, 'Accuracy': accuracy})
    
    def get_best_configuration(self):
        return max(self.results, key=lambda x: x['Accuracy'])
    
    def run(self, X_train, X_test, y_train, y_test):
        self.test_configurations(X_train, X_test, y_train, y_test)
        return self.get_best_configuration()
