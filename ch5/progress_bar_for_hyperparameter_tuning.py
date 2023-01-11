

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from ch4.dataset_class_final import DataSet

customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                       "number_customer_service_calls"],
                      file_name = "data/customer_churn_data.csv",
                      label_col = "churn",
                      pos_category = "yes"
                     ) 


parameters = {"max_depth":range(2, 8),
              "min_samples_leaf": range(5, 55, 5),
              "min_samples_split": range(10, 110, 5),
              "max_features": [2, 3],
              "n_estimators": [100, 150, 200, 250, 300, 350, 400]}


clf = RandomizedSearchCV(GradientBoostingClassifier(),
                         parameters,
                         n_jobs=4,
                         scoring = "roc_auc",
                         n_iter = 200,
                         random_state = 0,
                         verbose = 1)

clf.fit(customer_obj.train_features, customer_obj.train_labels)