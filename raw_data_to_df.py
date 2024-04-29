import pandas as pd
import pickle
import numpy as np
import os
import torch

import io

class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, n):
        if module == 'torch.storage' and n == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, n)

data = []
for root, dirs, files in os.walk(".\\play_data\\raw", topdown=False):
    for idx,name in enumerate(files):
        print(f"Appending file {idx}")
        with open(os.path.join(root, name), "rb") as file:
            # output = pickle.load(file)
            output = CPUUnpickler(file).load()
            for obs, probs, values in output:
                data.append({
                    "obs":obs,
                    "prob_0":probs[0],
                    "prob_1":probs[1],
                    "prob_2":probs[2],
                    "prob_3":probs[3],
                    "prob_4":probs[4],
                    "prob_5":probs[5],
                    "prob_6":probs[6],
                    "vals":values}
                )

df = pd.DataFrame(data)
df1 = df.iloc[:len(df)//2,:]
df2 = df.iloc[len(df)//2:,:]
df1.to_pickle('./play_data/gamestate_df1.pkl')
df2.to_pickle('./play_data/gamestate_df2.pkl')


# df = pd.read_pickle('./play_data/gamestate_df4.pkl')
# df1 = df.iloc[:len(df)//2,:]
# df2 = df.iloc[len(df)//2:,:]
# df1.to_pickle('./play_data/gamestate_df5.pkl')
# df2.to_pickle('./play_data/gamestate_df6.pkl')
