import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('Target', axis=1)
    y = data['Target']
    return preprocess_data(X), y

def create_ensemble_model(models):
    estimators = [(name, model) for name, model in models]
    ensemble = VotingClassifier(estimators=estimators, voting='hard')
    return ensemble
