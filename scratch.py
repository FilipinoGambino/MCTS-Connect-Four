from collections import defaultdict
import pandas as pd
import pickle
import os
import torch

# data = []
#
# for root, dirs, files in os.walk(".\\play_data\\run4", topdown=False):
#     for idx,name in enumerate(files):
#         print(f"Appending file {idx}")
#         with open(os.path.join(root, name), "rb") as file:
#             output = pickle.load(file)
#             for obs, probs, values in output:
#                 data.append({"obs":obs, "probs":torch.tensor(probs), "values":torch.tensor(values)})
#
# df = pd.DataFrame(data)
#
# print(df)
# df.to_pickle(".\\play_data\\run4\\gamestates_df.pkl")

df1 = pd.read_pickle(open(".\\play_data\\run1\\gamestates_df.pkl", "rb"))
df2 = pd.read_pickle(open(".\\play_data\\run2\\gamestates_df.pkl", "rb"))
df3 = pd.read_pickle(open(".\\play_data\\run3\\gamestates_df.pkl", "rb"))
df = pd.concat([df1, df2, df3], axis='index', ignore_index=True)
print(df)
print(df.describe())

df.to_pickle(".\\play_data\\gamestates_df.pkl")