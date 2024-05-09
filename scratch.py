from pathlib import Path
import os
import torch
import numpy as np
import pickle
import pandas as pd


df_paths = Path(f"{os.getcwd()}\\play_data")
df = None
for file in os.listdir(df_paths):
    fname = os.path.join(df_paths, file)
    if os.path.isdir(fname):
        continue
    if isinstance(df, pd.DataFrame):
        df = pd.concat([df, pd.read_pickle(fname)])
    else:
        df = pd.read_pickle(fname)
    print(f"{fname} loaded successfully")

df = df.reset_index(drop=True)
print(df.vals.unique())
print(df[df.vals == 1].vals.count())
print(df[df.vals == 0].vals.count())
print(df[df.vals == -1].vals.count())