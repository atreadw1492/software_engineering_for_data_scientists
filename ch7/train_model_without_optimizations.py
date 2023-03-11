
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

@profile
def train_logit_model(filename: str):

    ad_frame = pd.read_csv(filename)
    print("Finished reading dataset...\n\n")
    
    input_features = ad_frame[["site_category", "banner_pos"]]
    
    final_features = pd.concat([pd.get_dummies(input_features.site_category), 
                                ad_frame.banner_pos], axis = 1)
    

    logit_model = LogisticRegression(random_state = 0)
    print("Start training...\n\n")    
    logit_model.fit(final_features, ad_frame.click)
    
    return logit_model


start = time.time()
logit_model = train_logit_model("data/ads_train_data.csv")
end = time.time()
print("Finsihed running in ", end - start, " seconds")