

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

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

clf.fit(train_features, train_labels)