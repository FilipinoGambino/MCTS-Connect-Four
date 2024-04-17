from collections import defaultdict
import pandas as pd
import pickle
import os
from time import time
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
from itertools import count

game_idx = count()
print(next(game_idx))