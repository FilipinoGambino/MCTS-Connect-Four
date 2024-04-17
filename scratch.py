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

# df1 = pd.read_pickle(open(".\\play_data\\raw\\gamestates_df.pkl", "rb"))
# df2 = pd.read_pickle(open(".\\play_data\\gamestates_df.pkl", "rb"))
# df = pd.concat([df1, df2], axis='index', ignore_index=True)
# print(df)
# print(df.describe())
# df.to_pickle(".\\play_data\\gamestates_df.pkl")

df = pd.read_pickle('./play_data/gamestates_df1.pkl')

df1 = df.iloc[:10000,:]
# df2 = df.iloc[10000:20000,:]
# df3 = df.iloc[20000:30000,:]
# df4 = df.iloc[30000:40000,:]
# df5 = df.iloc[40000:50000,:]
# df6 = df.iloc[50000:,:]
df1.to_pickle(".\\play_data\\gamestates_df0.pkl")
# df2.to_pickle(".\\play_data\\gamestates_df2.pkl")
# df3.to_pickle(".\\play_data\\gamestates_df3.pkl")
# df4.to_pickle(".\\play_data\\gamestates_df4.pkl")
# df5.to_pickle(".\\play_data\\gamestates_df5.pkl")
# df6.to_pickle(".\\play_data\\gamestates_df6.pkl")