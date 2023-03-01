
import pandas as pd
import random
import time

def get_random_sample(filename: str, sample_rate: float):

    total_rows = sum(1 for line in open(filename, "rb")) - 1
    
    num_samples = round(total_rows * sample_rate)
    
    skip = sorted(random.sample(range(1, total_rows + 1),
                                total_rows - num_samples))
    
    return pd.read_csv(filename, skiprows=skip)
    

start = time.time()
sampled_data = get_random_sample("data/ads_train_data.csv", 0.1)
end = time.time()
print("Read in dataset in ", end - start, " seconds")