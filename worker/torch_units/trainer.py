from torch.utils.data import Dataset


class C4Dataset(Dataset):
    def __init__(self, game_data):
        self.df = game_data

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        game_state, probs, value = self.df.iloc[idx,:]
        return game_state, probs, value