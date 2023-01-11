
import stopit
from sklearn.ensemble import RandomForestClassifier
from ch4.dataset_class_final import DataSet

customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                       "number_customer_service_calls"],
                      file_name = "data/customer_churn_data.csv",
                      label_col = "churn",
                      pos_category = "yes"
                     ) 

with stopit.ThreadingTimeout(180) as context_manager:

    forest_model = RandomForestClassifier(n_estimators = 500).\
                                          fit(customer_obj.train_features, 
                                              customer_obj.train_labels)


# Did code finish running in under 180 seconds (3 minutes)?
if context_manager.state == context_manager.EXECUTED:
    print("FINISHED TRAINING MODEL...")

# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    
    # or raise an error if desired
    raise AssertionError("DID NOT FINISH MODEL TRAINING WITHIN TIME LIMIT")
