import torch
import pickle
import pandas as pd


df = pd.read_pickle("play_data/gamestate_df1.pkl")
print(df.iloc[:,1:].head(10))
