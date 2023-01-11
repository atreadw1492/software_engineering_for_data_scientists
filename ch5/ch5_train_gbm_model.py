

from sklearn.ensemble import GradientBoostingClassifier
from ch4.dataset_class_final import DataSet

customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                       "number_customer_service_calls"],
                      file_name = "data/customer_churn_data.csv",
                      label_col = "churn",
                      pos_category = "yes"
                     ) 

gbm_model = GradientBoostingClassifier(learning_rate = 0.1,
                                       n_estimators = 300,
                                       subsample = 0.7,
                                       min_samples_split = 40,
                                       max_depth = 3)

gbm_model.fit(customer_obj.train_features, customer_obj.train_labels)
