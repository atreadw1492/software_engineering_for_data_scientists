
import stopit
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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


with stopit.ThreadingTimeout(180) as context_manager:

    forest_model = RandomForestClassifier(n_estimators = 500).\
                                          fit(train_features, train_labels)


# Did code finish running in under 180 seconds (3 minutes)?
if context_manager.state == context_manager.EXECUTED:
    print("FINISHED TRAINING MODEL...")

# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    
    # or raise an error if desired
    raise AssertionError("DID NOT FINISH MODEL TRAINING WITHIN TIME LIMIT")
