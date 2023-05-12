
import joblib
import pandas as pd
import sys
sys.path.append("../../.")
sys.path.append("../")
import updated_offline_predictions_pipeline as pipeline

class Inputs:

    model = joblib.load("../../ch9/model_outputs/churn_model.sav")
    
    fake_config_file = "this_file_doesnt_exist.json"
    
    missing_key_config = {
                        "model_filename": "ch9/model_outputs/churn_model.sav",
                        "id_field": "customer_id",
                        "input_database": "data/customers.db",
                        "output_database": "data/customers.db",
                        "output_table": "customer_churn_predictions",
                        "input_features": ["total_day_minutes", 
                                           "total_day_calls",
                                           "number_customer_service_calls"]
                    }
    
    config_with_fake_input_file = {
                    "model_filename": "ch9/model_outputs/churn_model.sav",
                    "input_filename": "this_input_file_doesnt_exist.csv",
                    "id_field": "customer_id",
                    "input_database": "data/customers.db",
                    "output_database": "data/customers.db",
                    "output_table": "customer_churn_predictions",
                    "input_features": ["total_day_minutes", 
                                       "total_day_calls",
                                       "number_customer_service_calls"]
                }
    
    config_with_fake_model_file = {
                    "model_filename": "this_model_file_doesnt_exist.sav",
                    "input_filename": "data/customer_sample.csv",
                    "id_field": "customer_id",
                    "input_database": "data/customers.db",
                    "output_database": "data/customers.db",
                    "output_table": "customer_churn_predictions",
                    "input_features": ["total_day_minutes", 
                                       "total_day_calls",
                                       "number_customer_service_calls"]
                }    




def get_model_pred(inputs: dict, model) -> float:
    
    frame = pd.DataFrame.from_dict(inputs, orient = "index").transpose()
    
    return round(model.clf.predict_proba(frame)[0][1], 5)  


def test_input_file():
    
    customer_data = pd.read_csv("../../data/customer_churn_test_examples.csv")
    
    features = ['total_day_minutes', 'total_day_calls', 
                'num_customer_service_calls']
    
    for ix in range(customer_data.shape[0]):
        frame = customer_data[ix:ix+1]
        pred = Inputs.model.clf.\
                   predict_proba(frame[features])[0][1]
        
        
        assert round(pred, 5) == round(customer_data.pred.iloc[ix], 5)


def test_missing_config_keys():
    
    try:
        pipeline.run_config_checks(Inputs.missing_key_config)
    except AssertionError:
        return True
    
    return False

def test_fake_input_file():
    try:
        pipeline.\
        run_config_checks(Inputs.config_with_fake_input_file)
    except KeyError:
        return True
    
    return False


def test_fake_model_file():
    try:
        pipeline.\
        run_config_checks(Inputs.config_with_fake_model_file)
    except KeyError:
        return True
    
    return False
    