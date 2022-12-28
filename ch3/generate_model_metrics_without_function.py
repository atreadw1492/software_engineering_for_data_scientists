
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

forest_model = RandomForestClassifier(random_state = 0).fit(train_features,
                                                            train_labels)

logit_model = LogisticRegression().fit(train_features, train_labels)

boosting_model = GradientBoostingClassifier().fit(train_features, train_labels)


train_pred = forest_model.predict(train_features)
test_pred = forest_model.predict(test_features)

print("Train Precision = ", metrics.precision_score(train_labels, train_pred))
print("Train Recall = ", metrics.precision_score(train_labels, train_pred))
print("Train Accuracy = ", metrics.accuracy_score(train_labels, train_pred))


print("Train Precision = ", metrics.precision_score(test_labels, test_pred))
print("Train Recall = ", metrics.precision_score(test_labels, test_pred))
print("Train Accuracy = ", metrics.accuracy_score(test_labels, test_pred))


train_pred = logit_model.predict(train_features)
test_pred = logit_model.predict(test_features)


print("Train Precision = ", metrics.precision_score(train_labels, train_pred))
print("Train Recall = ", metrics.precision_score(train_labels, train_pred))
print("Train Accuracy = ", metrics.accuracy_score(train_labels, train_pred))


print("Train Precision = ", metrics.precision_score(test_labels, test_pred))
print("Train Recall = ", metrics.precision_score(test_labels, test_pred))
print("Train Accuracy = ", metrics.accuracy_score(test_labels, test_pred))

train_pred = boosting_model.predict(train_features)
test_pred = boosting_model.predict(test_features)

print("Train Precision = ", metrics.precision_score(train_labels, train_pred))
print("Train Recall = ", metrics.precision_score(train_labels, train_pred))
print("Train Accuracy = ", metrics.accuracy_score(train_labels, train_pred))


print("Train Precision = ", metrics.precision_score(test_labels, test_pred))
print("Train Recall = ", metrics.precision_score(test_labels, test_pred))
print("Train Accuracy = ", metrics.accuracy_score(test_labels, test_pred))
