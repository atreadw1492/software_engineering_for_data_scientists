

import pandas as pd
import sqlite3
from sklearn.ensemble import GradientBoostingClassifier
from ch4.dataset_class_final import DataSet


customer_data = pd.read_csv("data/customer_churn_data.csv")


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



output = pd.DataFrame([index for index in 
                       range(customer_obj.test_features.shape[0]) ],
                      columns=["customer_id"])

output["model_prediction"] = gbm_model.predict_proba(customer_obj.\
                                                     test_features)[::,1]


output["output_date"] = "2023-04-01"
    

sqliteConnection = sqlite3.connect('data/customers.db')
cursor = sqliteConnection.cursor()
print("Database created and Successfully Connected to SQLite")

insert_command = """insert into customer_churn_predictions(customer_id, 
                                model_prediction, prediction_date) 
                    values (?,?,?)"""
                    
cursor.executemany(insert_command, output.values)

sqliteConnection.commit()

sqlite_select_Query = "SELECT * FROM customer_churn_predictions"
cursor.execute(sqlite_select_Query)
record = cursor.fetchall()
cursor.close()



