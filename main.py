import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming your model classes are properly defined in their respective files within the 'models' directory
from models.knn_model import KNNModel
from models.logistic_regression_model import LogisticRegressionModel
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel

# Load your dataset
data = pd.read_csv('data.csv', delimiter=';')

# Identifying categorical and numerical columns
categorical_cols = ['Marital status', 'Application mode', 'Course', 'Daytime/evening attendance\t', 'Previous qualification', 'Nacionality', 'Mother\'s qualification', 'Father\'s qualification', 'Mother\'s occupation', 'Father\'s occupation', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']
numerical_cols = ['Previous qualification (grade)', 'Admission grade', 'Age at enrollment', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP']

# Splitting the dataset into the Training set and Test set
X = data.drop('Target', axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate and fit models

# KNN Model (Assuming KNNModel also integrates its preprocessing within the class in a similar fashion)
# If KNNModel does not yet integrate preprocessing, its instantiation and usage might need adjustments.
knn_model = KNNModel(X_train, X_test, y_train, y_test)
best_knn_config = knn_model.run()
print("Best KNN Configuration:", best_knn_config)

# Logistic Regression Model
# Adjust similarly if LogisticRegressionModel is updated to include preprocessing
lr_model = LogisticRegressionModel(X_train, X_test, y_train, y_test)
best_lr_config = lr_model.run()
print("Best Logistic Regression Configuration:", best_lr_config)

# SVM Model
# Adjust similarly if SVMModel is updated to include preprocessing
svm_model = SVMModel(X_train, X_test, y_train, y_test)
best_svm_config = svm_model.run()
print("Best SVM Configuration:", best_svm_config)

# Random Forest Model
# This has been updated to integrate preprocessing as demonstrated
rf_model = RandomForestModel(categorical_cols=categorical_cols, numerical_cols=numerical_cols)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)  # Now directly using score method for evaluation
