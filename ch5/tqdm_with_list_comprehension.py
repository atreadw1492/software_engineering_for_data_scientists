
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


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

forest_model = RandomForestClassifier(random_state = 0).fit(train_features,
                                                            train_labels)

logit_model = LogisticRegression().fit(train_features, train_labels)

boosting_model = GradientBoostingClassifier().fit(train_features, train_labels)

model_list = [forest_model, logit_model, boosting_model]


all_metrics = [get_metrics(model, train_features, test_features,
                           train_labels, test_labels) for model in
                           tqdm(model_list)]

