

import pandas as pd

@profile
def get_non_clicks(filename: str):

    ad_data = pd.read_csv(filename, chunksize = 10000000)
    ad_frame = ad_data.get_chunk()

    ad_frame = ad_frame[ad_frame.click == 0]

    return ad_frame

get_non_clicks("data/ads_data_sample.csv")
