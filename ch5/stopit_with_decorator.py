
from stopit import threading_timeoutable as timeoutable
from sklearn.ensemble import RandomForestClassifier
from ch4.dataset_class_final import DataSet

@timeoutable()
def train_model(features, labels):

    forest_model = RandomForestClassifier(n_estimators = 500).fit(features,
                                                                  labels)

    return forest_model



customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                       "number_customer_service_calls"],
                      file_name = "data/customer_churn_data.csv",
                      label_col = "churn",
                      pos_category = "yes"
                     ) 

forest_model = train_model(timeout = 180, 
                           features = customer_obj.train_features, 
                           labels = customer_obj.train_labels)


# Did code finish running in under 180 seconds (3 minutes)?
if forest_model:
    print("FINISHED TRAINING MODEL...")

# Did code timeout?
else:

    raise AssertionError("DID NOT FINISH MODEL TRAINING WITHIN TIME LIMIT")


# [Additional code block...]