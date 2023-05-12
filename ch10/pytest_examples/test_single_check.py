
import joblib
import pandas as pd
import sys
sys.path.append("../../.")
    
model = joblib.load("../../ch9/model_outputs/churn_model.sav")

def get_model_pred(inputs: dict, model) -> float:
    
    frame = pd.DataFrame.from_dict(inputs, orient = "index").transpose()
    
    return round(model.clf.predict_proba(frame)[0][1], 5)  


def test_1():
    
    inputs = {'total_day_minutes': 15,
             'total_day_calls': 1,
             'num_customer_service_calls': 0}
    
    assert get_model_pred(inputs, model) == 0.01994