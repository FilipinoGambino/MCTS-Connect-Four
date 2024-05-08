"""
Defines the actual model for making policy and value predictions given an observation.
"""

from pathlib import Path
import torch
from logging import getLogger
from types import SimpleNamespace

from agent.nns import create_model
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

    def __init__(self, flags: SimpleNamespace, device, fpath=None):
        self.flags = flags
        self.device = device
        self.api = None
        self.model = self.build_and_load_model(fpath)

    def get_pipes(self, n_pipes=1):
        """
        Creates a list of pipes on which observations of the game state will be listened for. Whenever
        an observation comes in, returns policy and value network predictions on that pipe.

        :param int n_pipes: number of pipes to create
        :return str(Connection): a list of all connections to the pipes that were created
        """
        if self.api is None:
            self.api = C4API(self)
            pipes = [self.api.create_pipe() for _ in range(n_pipes)]
            self.api.start()
            return pipes
        return [self.api.create_pipe() for _ in range(n_pipes)]

    def build_and_load_model(self, fpath):
        model = create_model(self.flags, device=self.device)
        if self.flags.worker_type == 'optimize':
            model.train()
        else:
            model.eval()
        model = model.share_memory()
        if fpath:
            checkpoint_state = torch.load(
                fpath,
                map_location=torch.device("cpu")
            )["model_state_dict"]

            model.load_state_dict(checkpoint_state)

        return model

    def save_model(self):
        fpath = Path(self.flags.model_dir) / Path(self.flags.nextgen_model_weight_fname)
        logger.info(f"Saving checkpoint to {fpath}")
        torch.save(obj={"model_state_dict": self.model.state_dict()}, f=fpath)

    def save_as_best_model(self):
        fpath = Path(self.flags.model_dir) / Path(self.flags.name + self.flags.best_model_weight_fname)
        logger.info(f"Saving checkpoint to {fpath}")
        torch.save(obj={"model_state_dict": self.model.state_dict()}, f=fpath)