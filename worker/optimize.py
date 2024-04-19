"""
Encapsulates the worker which trains ChessModels using game data from recorded games from a file.
"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from pathlib import Path
import pandas as pd
import torch
from torch.nn.functional import cross_entropy, mse_loss, softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

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
        self.data = OptimizeWorker.collect_data()

    def start(self):
        """
        Load the next generation model from disk and start doing the training endlessly.
        """
        model = C4Model(self.flags, self.flags.learner_device, self.flags.current_model_weight_fname)
        params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        logger.info(f"Training model with {params:,} parameters")

        optimizer = Adam(model.model.parameters(), **self.flags.optimizer_kwargs)

        def lr_lambda(epoch):
            min_pct = self.flags.min_lr_mod
            pct_complete = epoch / self.flags.max_epochs
            scaled_pct_complete = pct_complete * (1. - min_pct)
            return 1. - scaled_pct_complete

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.training(model, optimizer, lr_scheduler)

    def training(self, model, optimizer, lr_scheduler):
        """
        Does the actual training of the model, running it on game data endlessly.
        """
        max_epochs = self.flags.max_epochs

        train_ds = C4Dataset(self.data, self.flags.learner_device)
        train_dl = DataLoader(train_ds, batch_size=self.flags.batch_size, shuffle=True)
        if self.flags.enable_wandb:
            step = 0
            wandb.watch(model.model, self.flags.log_freq, log='all', log_graph=True)

        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1:>2}")
            for game_states, probs, values in train_dl:
                game_states = game_states
                outputs = model.model(game_states)

                policy_probs = softmax(outputs['policy_logits'], dim=-1).float()
                values = values.unsqueeze(-1).float()

                assert policy_probs.shape == probs.shape, f"policy probs: {policy_probs.shape} | probs: {probs.shape}"
                assert outputs['baseline'].shape == values.shape, f"policy values: {outputs['baseline'].shape} | values: {values.shape}"

                optimizer.zero_grad()

                prob_loss = cross_entropy(policy_probs, probs)
                val_loss = mse_loss(outputs['baseline'], values)

                total_loss = prob_loss + val_loss
                total_loss.backward()

                optimizer.step()
                if self.flags.enable_wandb:
                    stats = {
                        "Loss": {
                            "cross_entropy_loss": prob_loss.detach().item(),
                            "mse_loss": val_loss.detach().item(),
                            "total_loss": total_loss.detach().item()
                        },
                        "Learning Rate": lr_scheduler.get_last_lr()[0]
                    }
                    step += self.flags.batch_size
                    wandb.log(stats, step=step)
            lr_scheduler.step()
        model.save_model()

    @staticmethod
    def collect_data():
        df_path = Path(os.getcwd()) / Path("play_data") / Path("gamestate_df.pkl")
        return pd.read_pickle(df_path)
        # df_paths = Path(os.getcwd()) / Path("play_data")
        # df = None
        # for root, dirs, files in os.walk(df_paths, topdown=False):
        #     for file in files:
        #         fname = os.path.join(df_paths, file)
        #         if isinstance(df, pd.DataFrame):
        #             df = pd.read_pickle(fname)
        #         else:
        #             df = pd.concat([df, pd.read_pickle(fname)])
        # return df.reset_index(drop=True)