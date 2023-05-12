
import sys
sys.path.append(".")

import sqlite3
import argparse
import pandas as pd
import joblib
from datetime import datetime
import json


parser = argparse.ArgumentParser()
 
parser.add_argument("-c", "--config")

args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.load(config_file)
    

include_cols = config["input_features"] + [config["id_field"]]
new_data = pd.read_csv(config["input_filename"],
                       usecols = include_cols)

model = joblib.load(config["model_filename"])

output = pd.DataFrame()
output[config["id_field"]] = new_data[config["id_field"]].copy()
output["model_prediction"] = model.\
                             clf.\
                             predict_proba(new_data[config["input_features"]])[::,1]
                             
output["output_date"] = datetime.date(datetime.now()).strftime("%Y-%m-%d")
    

sqliteConnection = sqlite3.connect(config["output_database"])
cursor = sqliteConnection.cursor()
print("Connected to SQLite...")

table = config["output_table"]
id_field = config["id_field"]

insert_command = f"""INSERT INTO {table}({id_field}, 
                                 model_prediction, 
                                 prediction_date) 
                                 values (?,?,?)"""


                    
cursor.executemany(insert_command, output.values)

sqliteConnection.commit()
print("Committed model predictions...")

cursor.close()
sqliteConnection.close()
