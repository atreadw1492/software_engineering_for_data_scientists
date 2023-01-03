

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

customer_data = pd.read_csv("../data/customer_churn_data.csv")

train_data,test_data = train_test_split(customer_data, 
                                        train_size = 0.7,
                                        random_state = 0)


feature_list = ["total_day_minutes", "total_day_calls",
                "number_customer_service_calls"]

train_features = train_data[feature_list]
test_features = test_data[feature_list]

train_labels = train_data.churn.map(lambda key: 1 if key == "yes" else 0)
test_labels = test_data.churn.map(lambda key: 1 if key == "yes" else 0)


gbm_model = GradientBoostingClassifier(learning_rate = 0.1,
                                       n_estimators = 300,
                                       subsample = 0.7,
                                       min_samples_split = 40,
                                       max_depth = 3,
                                       verbose = 1
                                       )

gbm_model.fit(train_features, train_labels)