
import pandas as pd
from sklearn.linear_model import LogisticRegression

@profile
def train_logit_model(filename: str):

    ad_frame = pd.read_csv(filename)

    input_features = ad_frame[["site_category", "banner_pos"]]
    
    final_features = pd.concat([pd.get_dummies(input_features.site_category), 
                                ad_frame.banner_pos])
    
    logit_model = LogisticRegression(random_state = 0)
    
    logit_model.fit(ad_frame[final_features], ad_frame.click)
    
    return logit_model



logit_model = train_logit_model("data/ads_train_data.csv")