import torch
from torch.utils.data import Dataset


class C4Dataset(Dataset):
    def __init__(self, game_data):
        self.df = game_data

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]
        game_state = row.obs
        probs = torch.tensor([row.prob_0, row.prob_1, row.prob_2, row.prob_3, row.prob_4, row.prob_5, row.prob_6])
        value = torch.tensor(row.vals)
        return game_state, probs, value