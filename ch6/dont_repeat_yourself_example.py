
import pandas as pd
from tqdm import tqdm
import stopit


artists = pd.read_csv("data/artists.csv")

with stopit.ThreadingTimeout(60) as context_manager:

    for index in tqdm(range(artists.shape[0])):
        
        if artists.followers[index] > artists.followers.quantile(.999):
            print(artists.name[index])


# Did code finish running in under 180 seconds (3 minutes)?
if context_manager.state == context_manager.EXECUTED:
    print("FINISHED...")

# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    
    # or raise an error if desired
    print("DID NOT FINISH WITHIN TIME LIMIT")



with stopit.ThreadingTimeout(60) as context_manager:

    high_followers = artists.followers.quantile(.999)
    for index in tqdm(range(artists.shape[0])):
        
        if artists.followers[index] > high_followers:
            print(artists.name[index])


# Did code finish running in under 180 seconds (3 minutes)?
if context_manager.state == context_manager.EXECUTED:
    print("FINISHED...")

# Did code timeout?
elif context_manager.state == context_manager.TIMED_OUT:
    
    # or raise an error if desired
    print("DID NOT FINISH WITHIN TIME LIMIT")
