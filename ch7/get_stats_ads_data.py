
import pandas as pd



def get_stats(filename, field, max_chunks):

    ad_data = pd.read_csv(filename, 
                           chunksize=1000000,
                           usecols = [field])
  

    num_chunks = 0
    while num_chunks < max_chunks:
        
        frame = ad_data.get_chunk()
        
        if num_chunks == 0:
            counts = frame[field].value_counts()
        else:
            counts += frame[field].value_counts()
        
        num_chunks += 1
        print(num_chunks)
        
    return counts


get_stats("data/ads_train_data.csv", ["device_conn_type"], 5)
