
import pandas as pd
from tqdm import tqdm
import stopit


artists = pd.read_csv("data/artists.csv")

with stopit.ThreadingTimeout(60) as context_manager:

    high_followers = artists.followers.quantile(.999)
    for index in tqdm(range(artists.shape[0])):
        
        if artists.followers[index] > high_followers:
            print(artists.name[index])


if context_manager.state == context_manager.EXECUTED:
    print("FINISHED...")

elif context_manager.state == context_manager.TIMED_OUT:
    
    print("DID NOT FINISH WITHIN TIME LIMIT")
