from collections import defaultdict
import pandas as pd
import pickle
import os
import torch

data = []

for root, dirs, files in os.walk(".\\play_data\\run2", topdown=False):
    for name in files:
        with open(os.path.join(root, name), "rb") as file:
            output = pickle.load(file)
            for obs, probs, values in output:
                data.append({"obs":obs, "probs":torch.tensor(probs), "values":torch.tensor(values)})
                
df = pd.DataFrame(data)

print(df)
df.to_pickle(".\\play_data\\gamestates_df.pkl")

# df = pd.read_pickle(open(".\\play_data\\gamestates_df.pkl", "rb"))
# print(df.iloc[0,1])