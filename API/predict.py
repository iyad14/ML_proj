import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Add the parent directory of the 'models' package to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = FastAPI()

class StudentData(BaseModel):
    curricular_units_2nd_sem_approved: int
    curricular_units_2nd_sem_grade: float
    curricular_units_1st_sem_approved: int
    curricular_units_1st_sem_grade: float
    tuition_fees_up_to_date: int
    scholarship_holder: int
    curricular_units_2nd_sem_enrolled: int
    curricular_units_1st_sem_enrolled: int
    admission_grade: float
    displaced: int

# Load the trained models
knn_model_wrapper = joblib.load(r"C:\Users\USER\Desktop\ALIUSB\AliDor\Courses\10\COE546\New folder\4\ML_proj\models\SavedModel\KNN_model.pkl")
random_forest_model_wrapper = joblib.load(r"C:\Users\USER\Desktop\ALIUSB\AliDor\Courses\10\COE546\New folder\4\ML_proj\models\SavedModel\RandomForest_model.pkl")


@app.post("/predict/")
async def predict(student_data: StudentData):
    input_data = np.array([[value for value in student_data.dict().values()]])
    
    # Use the actual models to get predictions and probabilities
    knn_probabilities = knn_model_wrapper.model.predict_proba(input_data)[0]
    rf_probabilities = random_forest_model_wrapper.model.predict_proba(input_data)[0]

    # Getting top 2 predictions for each model
    knn_top2 = np.argsort(knn_probabilities)[-2:][::-1]
    rf_top2 = np.argsort(rf_probabilities)[-2:][::-1]

    return {
        "KNN Prediction": int(knn_model_wrapper.model.classes_[knn_top2[0]]),
        "KNN Confidence": float(knn_probabilities[knn_top2[0]]),
        "KNN 2nd Prediction": int(knn_model_wrapper.model.classes_[knn_top2[1]]),
        "KNN 2nd Confidence": float(knn_probabilities[knn_top2[1]]),
        "Random Forest Prediction": int(random_forest_model_wrapper.model.classes_[rf_top2[0]]),
        "Random Forest Confidence": float(rf_probabilities[rf_top2[0]]),
        "Random Forest 2nd Prediction": int(random_forest_model_wrapper.model.classes_[rf_top2[1]]),
        "Random Forest 2nd Confidence": float(rf_probabilities[rf_top2[1]]),
    }