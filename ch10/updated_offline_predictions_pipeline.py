import sys
sys.path.append(".")

import sqlite3
import argparse
import pandas as pd
import joblib
from datetime import datetime
import json
import os
from ch4.ml_model import MlModel


def run_pipeline():
    
    """Primary function of this script.  Calling this function 
    (with no parameters) will run the predictions pipeline."""
    
    config = parse_args()
    
    run_config_checks(config)
    new_data, model = fetch_data_and_model(config)
    output = get_data_to_upload(model, new_data, config)
    
    upload_to_database(config, output)


def parse_args() -> dict:

    """Extracts the config file input from the user.  
       Runs checks to make sure given config filename exists, and then returns
       the contents of the file"""    

    parser = argparse.ArgumentParser()
     
    parser.add_argument("-c", "--config")
    
    args = parser.parse_args()
  
    if not args.config:
        raise AssertionError("You must pass a config file when running this script")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError("""The config file you passed does not exist.
                                   Please check the path or spelling""")  
  
    with open(args.config, "r") as config_file:
        config = json.load(config_file)
        

    return config


def run_config_checks(config: dict) -> None:
    
    """Calling this function checks whether the input config file matches
       the required specifications for the customer churn model"""
    
    # run checks to make config file has what we need
    expected_config_keys = {"input_features", "id_field", "input_filename",
                        "model_filename", "input_features", "output_database",
                        "output_table"}

    both = set(config.keys()).intersection(expected_config_keys)
    if len(both) != len(expected_config_keys):
        raise AssertionError(f"""Config file must have all of these inputs: 
                             {expected_config_keys}""")
    
    if not os.path.exists(config["input_filename"]):
        raise KeyError("""Input filename doesn't exist.  
                             Make sure the path is correctly specified 
                             and spelled correctly""")

    if not os.path.exists(config["model_filename"]):
        raise KeyError("""Model filename doesn't exist.  
                             Make sure the path is correctly specified 
                             and spelled correctly""")




def fetch_data_and_model(config: dict) -> (pd.DataFrame, MlModel):

    """Takes a config object as input.  
       Uses the filenames set in the config to read in 
       and return the input dataset and model files"""    

    include_cols = config["input_features"] + [config["id_field"]]
    new_data = pd.read_csv(config["input_filename"],
                           usecols = include_cols)
    
    model = joblib.load(config["model_filename"])
    
    return new_data, model

def get_data_to_upload(model: MlModel, 
                       new_data: pd.DataFrame, 
                       config: dict) -> pd.DataFrame:

    """Creates and returns an output data frame containing predictions 
       from the input model and corresponding dataset (new_data)"""
    
    output = pd.DataFrame()
    output[config["id_field"]] = new_data[config["id_field"]].copy()
    output["model_prediction"] = model.\
                                 clf.\
                                 predict_proba(new_data[config["input_features"]])[::,1]
                                 
    output["output_date"] = datetime.date(datetime.now()).strftime("%Y-%m-%d")
    
    return output
        

def upload_to_database(config: dict, output: pd.DataFrame) -> None:

    """Uses specifications from the config object to upload the output data frame
       to a database"""    

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

    
if __name__ == "__main__":
    run_pipeline()




