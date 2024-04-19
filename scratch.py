import torch
import pickle
import pandas as pd


def check_type(x):
    if isinstance(x, dict):
        return {key:check_type(val) for key,val in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to("cuda:0")

df = pd.read_pickle("play_data/gamestate_df1.pkl")
gamestate = df.obs

t = check_type(gamestate[0])
print()

for key,val in t.items():
    print(f"{key}: {val}")