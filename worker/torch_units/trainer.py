import os
import pickle
from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from torchtnt.framework.unit import TrainUnit

Batch = Tuple[torch.tensor, torch.tensor]

class Trainer(TrainUnit[Batch]):
    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()
        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train_step(self, state: State, data: Batch) -> None:
        inputs, targets = data
        outputs = self.module(inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def on_train_epoch_end(self, state: State) -> None:
        # step the learning rate scheduler
        self.lr_scheduler.step()


class C4Dataset(Dataset):
    def __init__(self, game_data_dir, transform=None, target_transform=None):
        self.game_dir = game_data_dir
        self.tfm = transform
        self.target_tfm = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.game_dir, self.img_labels.iloc[idx, 0])
        game_state = read_image(img_path)
        victory = self.img_labels.iloc[idx, 1]
        if self.tfm:
            game_state = self.tfm(game_state)
        if self.target_tfm:
            victory = self.target_tfm(victory)
        return game_state, victory