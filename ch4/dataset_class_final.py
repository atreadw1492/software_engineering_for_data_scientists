

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics


class DataSet:

    def __init__(self,
                 feature_list: list,
                 file_name: str,
                 label_col: str,
                 pos_category: str):

        self.customer_data = pd.read_csv(file_name)

        self.train_data,self.test_data = train_test_split(self.customer_data,
                                                 train_size = 0.7,
                                                 random_state = 0)


        self.train_data = self.train_data.reset_index(drop = True)
        self.test_data = self.test_data.reset_index(drop = True)


        self.feature_list = feature_list

        self.train_features = self.train_data[feature_list]
        self.test_features = self.test_data[feature_list]

        self.train_labels = self.train_data[label_col].\
                                 map(lambda key: 1 if key == pos_category 
                                     else 0)
                                 
        self.test_labels = self.test_data[label_col].\
                                map(lambda key: 1 if key == pos_category 
                                    else 0)
        
        self.pos_category = pos_category
        self.label_col = label_col

    def get_summary_plots(self):
        
        for feature in self.feature_list:
            self.train_data[feature].hist()
            plt.title(feature)
            plt.show()
            
            
    def get_model_metrics(self, train_pred: pd.Series, test_pred: pd.Series):
        
        print("Train precision = ", 
              metrics.precision_score(self.train_labels,
                                      train_pred))
        
        print("Test precision = ", 
              metrics.precision_score(self.test_labels,
                                      test_pred))
    
        
        
        