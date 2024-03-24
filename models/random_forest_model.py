from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = []

    def test_configurations(self, n_estimators_list=[100], max_depth_list=[None]):
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                self.results.append({'N_estimators': n_estimators, 'Max_depth': max_depth, 'Accuracy': accuracy})
    
    def get_best_configuration(self):
        return max(self.results, key=lambda x: x['Accuracy'])
    
    def run(self):
        self.test_configurations()
        return self.get_best_configuration()
