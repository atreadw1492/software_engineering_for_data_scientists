

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from ch4.dataset_class_final import DataSet


def get_metrics(model, train_features, test_features,
                train_labels, test_labels):

    train_pred = model.predict(train_features)
    test_pred = model.predict(test_features)

    train_precision = metrics.precision_score(train_labels, train_pred)
    train_recall = metrics.recall_score(train_labels, train_pred)
    train_accuracy = metrics.accuracy_score(train_labels, train_pred)

    test_precision = metrics.precision_score(test_labels, test_pred)
    test_recall = metrics.recall_score(test_labels, test_pred)
    test_accuracy = metrics.accuracy_score(test_labels, test_pred)

    return train_precision, train_recall, train_accuracy, test_precision, test_recall, test_accuracy


customer_obj = DataSet(feature_list = ["total_day_minutes", "total_day_calls",
                       "number_customer_service_calls"],
                      file_name = "data/customer_churn_data.csv",
                      label_col = "churn",
                      pos_category = "yes"
                     )

forest_model = RandomForestClassifier(random_state = 0).fit(customer_obj.train_features,
                                                            customer_obj.train_labels)

logit_model = LogisticRegression().fit(customer_obj.train_features, customer_obj.train_labels)

boosting_model = GradientBoostingClassifier().fit(customer_obj.train_features, customer_obj.train_labels)

model_list = [forest_model, logit_model, boosting_model]


all_metrics = [get_metrics(model, customer_obj.train_features, customer_obj.test_features,
                           customer_obj.train_labels, customer_obj.test_labels) for model in
                           tqdm(model_list)]

