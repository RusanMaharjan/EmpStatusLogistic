# API Code
import pandas as pd
from app.model import load_model_scaler
from app.schema import EmployeeStatus
from fastapi import FastAPI

app = FastAPI()
model, scaler = load_model_scaler()

@app.get("/")
def home():
    return "Welcome to Fast API Employee Status Project!!"

@app.post("/predict-status")
def predict_status(data: EmployeeStatus):
    input_data = pd.DataFrame([
        data.model_dump()
    ])
    input_scaler = scaler.transform(input_data)
    prediction = model.predict(input_scaler)[0]
    return {
        "Predicted_Status": int(prediction),
        "Status": "Likely to be Terminated!!" if prediction == 1 else "Likely to be Active!!"
    }


