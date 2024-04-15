"""
Encapsulates the worker which trains ChessModels using game data from recorded games from a file.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from pathlib import Path
import pandas as pd
import torch
from torch.nn.functional import cross_entropy, mse_loss, softmax
from torch.optim import Adam
from torch.utils.data import DataLoader

from agent.c4_model import C4Model
from .torch_units.trainer import C4Dataset

logger = getLogger(__name__)

def start(flags):
    """
    Helper method which just kicks off the optimization using the specified config
    :param flags: config to use
    """
    return OptimizeWorker(flags).start()

class OptimizeWorker:
    """
    Worker which optimizes a ChessModel by training it on game data

    Attributes:
        :ivar Config config: config for this worker
        :ivar ChessModel model: model to train
        :ivar dequeue,dequeue,dequeue dataset: tuple of dequeues where each dequeue contains game states,
            target policy network values (calculated based on visit stats
                for each state during the game), and target value network values (calculated based on
                    who actually won the game after that state)
        :ivar ProcessPoolExecutor executor: executor for running all of the training processes
    """
    def __init__(self, flags):
        self.flags = flags
        self.data = pd.read_pickle(open(".\\play_data\\gamestates_df.pkl", "rb"))

    def start(self):
        """
        Load the next generation model from disk and start doing the training endlessly.
        """
        model = C4Model(self.flags).model
        optimizer = Adam(model.parameters())

        t = self.flags.unroll_length
        b = self.flags.batch_size

        def lr_lambda(epoch):
            min_pct = self.flags.min_lr_mod
            pct_complete = min(epoch * t * b, self.flags.total_steps) / self.flags.total_steps
            scaled_pct_complete = pct_complete * (1. - min_pct)
            return 1. - scaled_pct_complete

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.training(model, optimizer, lr_scheduler)

    def training(self, model, optimizer, lr_scheduler):
        """
        Does the actual training of the model, running it on game data endlessly.
        """
        max_epochs = self.flags.max_epochs

        train_ds = C4Dataset(self.data)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        for epoch in range(max_epochs):
            for game_states, probs, values in train_dl:
                outputs = model(game_states)

                policy_probs = softmax(outputs['policy_logits'], dim=-1)
                for prob in probs:
                    print(prob.shape)
                print(values.shape)

                prob_loss = cross_entropy(policy_probs, probs)
                val_loss = mse_loss(outputs['baseline'], values)

                total_loss = prob_loss + val_loss
                total_loss.backward()

                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()