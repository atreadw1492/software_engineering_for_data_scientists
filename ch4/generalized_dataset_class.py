

import pandas as pd
from sklearn.model_selection import train_test_split


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