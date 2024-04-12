from collections import defaultdict
import pandas as pd
import pickle
import os

data = []

for root, dirs, files in os.walk(".\\play_data", topdown=False):
    for name in files:
        with open(os.path.join(root, name), "rb") as file:
            output = pickle.load(file)
            for obs, probs, win in output:
                data.append({"obs":obs, "probs":probs, "win":win}, ignore_index=True)
        break
print(df)