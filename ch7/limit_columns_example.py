
import pandas as pd

ad_data = pd.read_csv("data/ads_train_data.csv", 
                      chunksize=1000000,
                      usecols = ["id", "hour", "C1", "banner_pos"])

ad_frame = ad_data.get_chunk()