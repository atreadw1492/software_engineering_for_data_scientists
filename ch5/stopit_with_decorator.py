
from stopit import threading_timeoutable as timeoutable
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@timeoutable()
def train_model(features, labels):

    forest_model = RandomForestClassifier(n_estimators = 500).fit(features,
                                                                  labels)

    return forest_model

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


forest_model = train_model(timeout = 180, 
                           features = train_features, 
                           labels = train_labels)


# Did code finish running in under 180 seconds (3 minutes)?
if forest_model:
    print("FINISHED TRAINING MODEL...")

# Did code timeout?
else:

    raise AssertionError("DID NOT FINISH MODEL TRAINING WITHIN TIME LIMIT")


# [Additional code block...]