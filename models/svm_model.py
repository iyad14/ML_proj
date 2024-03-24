from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVMModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []

    def test_configurations(self, C_values=[1.0], kernel_types=['linear', 'rbf']):
        for C in C_values:
            for kernel in kernel_types:
                model = SVC(C=C, kernel=kernel)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                self.results.append({'C': C, 'Kernel': kernel, 'Accuracy': accuracy})
    
    def get_best_configuration(self):
        return max(self.results, key=lambda x: x['Accuracy'])
    
    def run(self):
        self.test_configurations()
        return self.get_best_configuration()
