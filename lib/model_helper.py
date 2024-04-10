"""
Helper methods for working with trained models.
"""
from logging import getLogger
import os
from pathlib import Path
import torch

logger = getLogger(__name__)


def load_best_model_weight(model):
    """
    :param chess_zero.agent.model.ChessModel model:
    :return:
    """
    file_path = os.path.join(os.getcwd(), model.flags.model_dir, model.flags.checkpoint_file)
    return torch.load(file_path, map_location=torch.device("cpu"))


def save_as_best_model(model, idx):
    """
    :param chess_zero.agent.model.ChessModel model:
    :param int idx:
    :return:
    """
    file_path = os.path.join(os.getcwd(), model.flags.model_dir, model.flags.best_model_weight_fname % idx)
    torch.save(
        obj={"model_state_dict": model.model.state_dict()},
        f=file_path
    )