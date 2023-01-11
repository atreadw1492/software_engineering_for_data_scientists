


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CustomerData:

    def __init__(self,
                 feature_list):

        self.customer_data = pd.read_csv("../data/customer_churn_data.csv")

        self.train_data,self.test_data = train_test_split(self.customer_data,
                                                 train_size = 0.7,
                                                 random_state = 0)

        self.train_data = self.train_data.reset_index(drop = True)
        self.test_data = self.test_data.reset_index(drop = True)

        self.feature_list = feature_list

        self.train_features = self.train_data[feature_list]
        self.test_features = self.test_data[feature_list]

        self.train_labels = self.train_data.churn.map(lambda key: 1 if key == "yes" else 0)
        self.test_labels = self.test_data.churn.map(lambda key: 1 if key == "yes" else 0)

    def get_summary_plots(self):
        
        for feature in self.feature_list:
            self.train_data[feature].hist()
            plt.title(feature)
            plt.show()
            
    
    def train_rules_model(self):
        
        self.train_data["high_service_calls"] = self.train_data.number_customer_service_calls.map(lambda val: 1 if val > 3 else 0)
        self.train_data["has_international_plan"] = self.train_data.international_plan.map(lambda val: 1 if val == "yes" else 0)
        
        self.train_data["rules_pred"] = [max(calls, plan) for calls,plan in zip(self.train_data.high_service_calls, self.train_data.has_international_plan)]
        
        
        
        