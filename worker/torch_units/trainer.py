import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Union


class C4Dataset(Dataset):
    def __init__(self, game_data, device):
        self.df = game_data
        self.device = device

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]
        game_state = self._to_tensor(row.obs)
        probs = self._to_tensor([row.prob_0, row.prob_1, row.prob_2, row.prob_3, row.prob_4, row.prob_5, row.prob_6])
        value = self._to_tensor(row.vals)
        return game_state, probs, value

    def _to_tensor(self, x: Union[Dict, torch.Tensor]) -> Dict[str, Union[Dict, torch.Tensor]]:
        if isinstance(x, dict):
            return {key: self._to_tensor(val) for key, val in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, (list, np.int64)):
            return torch.tensor(x).to(self.device)
        else:
            raise TypeError(f"Expected dict, torch.Tensor, list, or np.int64, but got: {type(x)}")