from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Allow any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./trainedModel2.h5"
binary_model = load_model(model_path)

class Item(BaseModel):
    gender: str
    age: int
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str
    
@app.get("/ping")
async def ping():
    return "Hello, I am alive"
    
@app.post("/predict")
def predict(item: Item):
    # Map categorical variables to numerical values
    gender_mapping = {'Male': 0, 'Female': 1, 'Other': 2}
    ever_married_mapping = {'No': 0, 'Yes': 1}
    work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_type_mapping = {'Urban': 0, 'Rural': 1}
    smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

    # Convert input data to a NumPy array
    input_data = np.array([[
        gender_mapping.get(item.gender, 2),
        item.age, item.hypertension, item.heart_disease,
        ever_married_mapping.get(item.ever_married, 2),
        work_type_mapping.get(item.work_type, 4),
        residence_type_mapping.get(item.Residence_type, 1),
        item.avg_glucose_level, item.bmi,
        smoking_status_mapping.get(item.smoking_status, 3)
    ]])

    # Make prediction using the loaded model
    prediction = binary_model.predict(input_data)

    # The output is a probability, you can convert it to a class (0 or 1) based on a threshold
    threshold = 0.5
    prediction_message = "Positive" if prediction[0, 0] > threshold else "Negative"

    return {"prediction": prediction_message}
    
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
