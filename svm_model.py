from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMModel:
    def __init__(self):
        self.results = []

    def test_configurations(self, X_train, X_test, y_train, y_test, C_values=[1.0], kernel_types=['linear', 'rbf']):
        for C in C_values:
            for kernel in kernel_types:
                model = SVC(C=C, kernel=kernel)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                self.results.append({'C': C, 'Kernel': kernel, 'Accuracy': accuracy})
    
    def get_best_configuration(self):
        return max(self.results, key=lambda x: x['Accuracy'])
    
    def run(self, X_train, X_test, y_train, y_test):
        self.test_configurations(X_train, X_test, y_train, y_test)
        return self.get_best_configuration()
