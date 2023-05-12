
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
    
    
    def test_1(self):
        
        inputs = {'total_day_minutes': 15,
                 'total_day_calls': 1,
                 'num_customer_service_calls': 0}
        
        self.assertEqual(self.get_model_pred(inputs, TestModelOutputs.model),
                         0.01994)
        

if __name__ == '__main__':
    unittest.main()