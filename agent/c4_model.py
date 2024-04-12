"""
Defines the actual model for making policy and value predictions given an observation.
"""

from pathlib import Path
import torch
from logging import getLogger
from types import SimpleNamespace

from .nns import create_model
from agent.c4_api import C4API


logger = getLogger(__name__)


class C4Model:
    """
    The model which can be trained to take observations of a game of chess and return value and policy
    predictions.

    Attributes:
        :ivar Config config: configuration to use
        :ivar Model model: the PyTorch model to use for predictions
        :ivar ChessModelAPI api: the api to use to listen for and then return this models predictions (on a pipe).
    """

    def __init__(self, flags: SimpleNamespace):
        self.flags = flags
        self.api = None
        self.model = self.build_and_load_best_model()

    def get_pipes(self, n_pipes=1):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int n_pipes: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        if self.api is None:
            self.api = C4API(self)
            self.api.start()
        return [self.api.create_pipe() for _ in range(n_pipes)]

    def build_and_load_best_model(self):
        model = create_model(self.flags, device=self.flags.device)
        model.eval()
        model = model.share_memory()

        checkpoint_state = torch.load(
            Path(self.flags.load_dir) / self.flags.checkpoint_file, map_location=torch.device("cpu")
        )

        model.load_state_dict(checkpoint_state["model_state_dict"])
        return model

    def save_checkpoint(self, checkpoint_path, weight_path, step, total_games_played):
        """

        :param str checkpoint_path: path to save the entire configuration to
        :param str weight_path: path to save the model weights to
        :param str step: current step through all games played so far
        :param str total_games_played: current number of games played so far
        """
        logger.debug(f"Saving checkpoint to {checkpoint_path}")
        torch.save({"model_state_dict": self.model.state_dict(),
                        "step": step,
                        "total_games_played": total_games_played},
                   checkpoint_path + ".pt",)
        torch.save({"model_state_dict": self.model.state_dict()},
                   weight_path + ".pt")