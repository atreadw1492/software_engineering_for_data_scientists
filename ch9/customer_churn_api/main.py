
# main.py

import sys
sys.path.append("../../")

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib



class ModelFeatures(BaseModel):
    total_day_minutes: float
    total_day_calls: int
    num_customer_service_calls: int
    
model = joblib.load("../model_outputs/churn_model.sav")

app = FastAPI()

@app.post("/churn/")
async def get_churn_predictions(features: ModelFeatures):
    
    
    model_inputs = pd.DataFrame([features.total_day_calls,
                    features.total_day_minutes,
                    features.num_customer_service_calls]).transpose()
    
    pred = model.clf.predict_proba(model_inputs)[0,1]
    
    return {"prob_of_churn": pred}