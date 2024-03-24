import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import your model classes
from models.knn_model import KNNModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel

# Assuming 'data.csv' is your dataset and 'target' is your target column
data = pd.read_csv('data.csv', delimiter=';')

# Preprocessing
# Identifying categorical and numerical columns
categorical_cols = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
numerical_cols = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP']

# Handling missing values
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundling preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Splitting the dataset into the Training set and Test set
X = data.drop('Target', axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

 # KNN Model
knn_model = KNNModel()
best_knn_config = knn_model.run(X_train, X_test, y_train, y_test)
print("Best KNN Configuration:", best_knn_config)

# Logistic Regression Model
lr_model = LogisticRegressionModel()
best_lr_config = lr_model.run(X_train, X_test, y_train, y_test)
print("Best Logistic Regression Configuration:", best_lr_config)

# SVM Model
svm_model = SVMModel()
best_svm_config = svm_model.run(X_train, X_test, y_train, y_test)
print("Best SVM Configuration:", best_svm_config)

# Random Forest Model
rf_model = RandomForestModel()
best_rf_config = rf_model.run(X_train, X_test, y_train, y_test)
print("Best Random Forest Configuration:", best_rf_config)
