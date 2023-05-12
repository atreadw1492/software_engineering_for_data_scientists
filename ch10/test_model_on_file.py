
import unittest
import joblib
import pandas as pd
import sys
sys.path.append(".")


class TestModelOutputs(unittest.TestCase):
    
    model = joblib.load("ch9/model_outputs/churn_model.sav")
    
    def get_model_pred(self, inputs: dict, model) -> float:
        
        frame = pd.DataFrame.from_dict(inputs, orient = "index").transpose()
        
        return round(model.clf.predict_proba(frame)[0][1], 5)  
    
    
    def test_input_file(self):
        
        customer_data = pd.read_csv("data/customer_churn_test_examples.csv")
        
        features = ['total_day_minutes', 'total_day_calls', 
                    'num_customer_service_calls']
        
        #pred = TestModelOutputs.model.clf.predict_proba(customer_data[features])
        
        for ix in range(customer_data.shape[0]):
            frame = customer_data[ix:ix+1]
            pred = TestModelOutputs.model.clf.\
                       predict_proba(frame[features])[0][1]
            
        
            self.assertEqual(round(pred, 5), 
                             round(customer_data.pred.iloc[ix], 5))
        

if __name__ == '__main__':
    unittest.main()