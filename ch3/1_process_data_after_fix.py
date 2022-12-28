import pandas as pd
import numpy as np

num_states = 50


def process_data(file_name, num_samples):

    df = pd.read_csv(file_name)

    df = df.fillna(df.median())

    if num_samples > 0:
        df = df.sample(num_samples)
    
    return df
