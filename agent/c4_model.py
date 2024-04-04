"""
Defines the actual model for making policy and value predictions given an observation.
"""

from pathlib import Path
import torch
from logging import getLogger
from types import SimpleNamespace

from nns import create_model
from agent.c4_api import C4API


logger = getLogger(__name__)


class C4Model:
    """
    The model which can be trained to take observations of a game of chess and return value and policy
    predictions.

    Attributes:
        :ivar Config config: configuration to use
        :ivar Model model: the PyTorch model to use for predictions
        :ivar digest: basically just a hash of the file containing the weights being used by this model
        :ivar ChessModelAPI api: the api to use to listen for and then return this models predictions (on a pipe).
    """

    def __init__(self, flags: SimpleNamespace, is_actor=False):
        self.flags = flags
        self.digest = None
        self.api = None
        self.is_actor = is_actor

        self.model = None
        self.optimizer = None
        self.scheduler = None

    def get_pipes(self, num=1):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int num: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        if self.api is None:
            self.api = C4API(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(num)]

    def build(self):
        self.model = create_model(self.flags, device=self.flags.device)

        if self.is_actor:
            self.model.eval()
            self.model = self.model.share_memory()

            self.optimizer = self.flags.optimizer_class(
                self.model.parameters(),
                **self.flags.optimizer_kwargs
            )

            n_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f'Training model with {n_trainable_params:,d} parameters.')

        else:
            self.model.train()
            self.model = self.model.share_memory()


        t = self.flags.unroll_length
        b = self.flags.batch_size

        def lr_lambda(epoch):
            min_pct = self.flags.min_lr_mod
            pct_complete = min(epoch * t * b, self.flags.total_steps) / self.flags.total_steps
            scaled_pct_complete = pct_complete * (1. - min_pct)
            return 1. - scaled_pct_complete

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if self.flags.load_dir and not self.flags.weights_only:
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_state = torch.load(
            Path(self.flags.load_dir) / self.flags.checkpoint_file, map_location=torch.device("cpu")
        )

        self.model.load_state_dict(checkpoint_state["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])

    def save(self, checkpoint_path, weight_path, step, total_games_played):
        """

        :param str checkpoint_path: path to save the entire configuration to
        :param str weight_path: path to save the model weights to
        :param str step: current step through all games played so far
        :param str total_games_played: current number of games played so far
        """
        logger.debug(f"Saving checkpoint to {checkpoint_path}")
        torch.save({"model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "step": step,
                        "total_games_played": total_games_played},
                   checkpoint_path + ".pt",)
        torch.save({"model_state_dict": self.model.state_dict()},
                   weight_path + ".pt")