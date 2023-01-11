
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:

    def __init__(self,
                 feature_list,
                 file_name,
                 label):

        self.data = pd.read_csv(file_name)
        self.file_name = file_name

        self.train_data,self.test_data = train_test_split(self.customer_data,
                                                 train_size = 0.7,
                                                 random_state = 0)


        self.feature_list = feature_list

        self.train_features = self.train_data[feature_list]
        self.test_features = self.test_data[feature_list]

        self.train_labels = self.train_data[label].map(lambda key: 1 if key == "yes" else 0)
        self.test_labels = self.test_data[label].map(lambda key: 1 if key == "yes" else 0)
