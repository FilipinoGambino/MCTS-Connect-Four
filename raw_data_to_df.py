from collections import defaultdict
import pandas as pd
import pickle
import os
from time import time
import torch

# data = []
# for root, dirs, files in os.walk(".\\play_data\\raw", topdown=False):
#     for idx,name in enumerate(files):
#         print(f"Appending file {idx}")
#         with open(os.path.join(root, name), "rb") as file:
#             output = pickle.load(file)
#             for obs, probs, values in output:
#                 data.append({
#                     "obs":obs,
#                     "prob_0":probs[0],
#                     "prob_1":probs[1],
#                     "prob_2":probs[2],
#                     "prob_3":probs[3],
#                     "prob_4":probs[4],
#                     "prob_5":probs[5],
#                     "prob_6":probs[6],
#                     "vals":values}
#                 )
#
# df = pd.DataFrame(data)
# df.to_pickle('./play_data/gamestate_df.pkl')


df = pd.read_pickle('./play_data/gamestate_df.pkl')
obs = df['obs']
probs = df.apply(lambda row: [row.prob_0, row.prob_1, row.prob_2, row.prob_3, row.prob_4, row.prob_5, row.prob_6], axis=1)
vals = df['vals']
print(obs.head(10))
print(probs.head(10))
print(vals.head(10))
print(probs[0])