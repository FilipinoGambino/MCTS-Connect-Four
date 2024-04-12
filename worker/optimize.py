"""
Encapsulates the worker which trains ChessModels using game data from recorded games from a file.
"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from pathlib import Path
from random import shuffle
import torch
from torch.optim import Adam

import numpy as np

from agent.c4_model import C4Model
from lib.data_helper import get_game_data_filenames, read_game_data_from_file, get_next_generation_model_dirs
from lib.model_helper import load_best_model_weight
from torch_units.trainer import Trainer

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
        self.fit_unit = self.load_torch_training_unit()
        self.dataset = deque(),deque(),deque()
        self.executor = ProcessPoolExecutor(max_workers=flags.trainer.cleaning_processes)
        self.fnames = deque()

    def start(self):
        """
        Load the next generation model from disk and start doing the training endlessly.
        """
        self.training()

    def training(self):
        """
        Does the actual training of the model, running it on game data endlessly.
        """
        self.fnames.extend(get_game_data_filenames(self.flags.resource))
        shuffle(self.fnames)
        total_steps = self.flags.trainer.start_total_steps

        while True:
            self.fill_queue()
            steps = self.train_epoch(self.flags.trainer.epoch_to_checkpoint)
            total_steps += steps
            self.save_current_model()
            a, b, c = self.dataset
            while len(a) > self.flags.trainer.dataset_size/2:
                a.popleft()
                b.popleft()
                c.popleft()

    def train_epoch(self, epochs):
        """
        Runs some number of epochs of training
        :param int epochs: number of epochs
        :return: number of datapoints that were trained on in total
        """
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        self.model.model.fit(state_ary, [policy_ary, value_ary],
                             batch_size=self.flags.batch_size,
                             epochs=epochs,
                             shuffle=True,
                             validation_split=0.02)
        steps = (state_ary.shape[0] // self.flags.batch_size) * epochs
        return steps

    def load_best_model(self, model, optimizer):
        checkpoint_state = torch.load(
            Path(self.flags.load_dir) / self.flags.checkpoint_file, map_location=torch.device("cpu")
        )

        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    def save_current_model(self):
        """
        Saves the current model as the next generation model to the appropriate directory
        """
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        model_dir = os.path.join(self.flags.next_gen_model_dir, self.flags.next_gen_model_dirname_tmpl % model_id)
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, self.flags.next_gen_model_config_filename)
        weight_path = os.path.join(model_dir, self.flags.next_gen_model_weight_filename)
        self.model.save(config_path, weight_path)

    def fill_queue(self):
        """
        Fills the self.dataset queues with data from the training dataset.
        """
        futures = deque()
        with ProcessPoolExecutor(max_workers=self.flags.trainer.cleaning_processes) as executor:
            for _ in range(self.flags.trainer.cleaning_processes):
                if len(self.fnames) == 0:
                    break
                filename = self.fnames.popleft()
                logger.debug(f"loading data from {filename}")
                futures.append(executor.submit(load_data_from_file,filename))
            while futures and len(self.dataset[0]) < self.flags.trainer.dataset_size:
                for x,y in zip(self.dataset,futures.popleft().result()):
                    x.extend(y)
                if len(self.fnames) > 0:
                    filename = self.fnames.popleft()
                    logger.debug(f"loading data from {filename}")
                    futures.append(executor.submit(load_data_from_file,filename))

    def collect_all_loaded_data(self):
        """

        :return: a tuple containing the data in self.dataset, split into
        (state, policy, and value).
        """
        state_ary, policy_ary, value_ary = self.dataset

        state_ary1 = np.asarray(state_ary, dtype=np.float32)
        policy_ary1 = np.asarray(policy_ary, dtype=np.float32)
        value_ary1 = np.asarray(value_ary, dtype=np.float32)
        return state_ary1, policy_ary1, value_ary1

    def load_torch_training_unit(self):
        """
        Loads the next generation model from the appropriate directory. If not found, loads
        the best known model.
        """
        model = C4Model(self.flags)
        optimizer = Adam(model.model.parameters())

        t = self.flags.unroll_length
        b = self.flags.batch_size

        def lr_lambda(epoch):
            min_pct = self.flags.min_lr_mod
            pct_complete = min(epoch * t * b, self.flags.total_steps) / self.flags.total_steps
            scaled_pct_complete = pct_complete * (1. - min_pct)
            return 1. - scaled_pct_complete

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        self.load_best_model(model, optimizer)

        return Trainer(model.model, optimizer, lr_scheduler)


def load_data_from_file(filename):
    data = read_game_data_from_file(filename)
    return convert_to_cheating_data(data)


def convert_to_cheating_data(data):
    """
    :param data: format is SelfPlayWorker.buffer
    :return:
    """
    state_list = []
    policy_list = []
    value_list = []
    for state_fen, policy, value in data:

        state_planes = canon_input_planes(state_fen)

        if is_black_turn(state_fen):
            policy = Config.flip_policy(policy)

        move_number = int(state_fen.split(' ')[5])
        value_certainty = min(5, move_number)/5 # reduces the noise of the opening... plz train faster
        sl_value = value*value_certainty + testeval(state_fen, False)*(1-value_certainty)

        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(sl_value)

    return np.asarray(state_list, dtype=np.float32), np.asarray(policy_list, dtype=np.float32), np.asarray(value_list, dtype=np.float32)