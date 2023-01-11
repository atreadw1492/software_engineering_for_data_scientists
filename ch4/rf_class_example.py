
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from dataset_class_final import DataSet

class RandomForestModel:

    def __init__(self, 
                 parameters: dict, 
                 n_jobs: int, 
                 scoring: str, 
                 n_iter: int, 
                 random_state: int):

        self.parameters = parameters
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.n_iter = n_iter
        self.random_state = random_state

    def tune(self, X_features, y):

        self.clf = RandomizedSearchCV(RandomForestClassifier(),
                                      self.parameters,
                                      n_jobs=self.n_jobs,
                                      scoring = self.scoring,
                                      n_iter = self.n_iter,
                                      random_state = self.random_state)

        self.clf.fit(X_features, y)

    def predict(self, X_features):

        return self.clf.predict(X_features)
    

customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                        "number_customer_service_calls"],
                       file_name = "../data/customer_churn_data.csv",
                       label_col = "churn",
                       pos_category = "yes"
                      )


parameters = {"max_depth":range(2, 6),
              "min_samples_leaf": range(5, 55, 5),
              "min_samples_split": range(10, 110, 5),
              "max_features": [2, 3, 4, 5, 6],
              "n_estimators": [50, 100, 150, 200]}


forest = RandomForestModel(parameters = parameters,
                           n_jobs = 4,
                           scoring = "roc_auc",
                           n_iter = 10,
                           random_state = 0)

forest.tune(customer_obj.train_features, customer_obj.train_labels)



