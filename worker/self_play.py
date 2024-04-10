from collections import deque
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
import os
from threading import Thread
from time import time
from types import SimpleNamespace

from c4_gym import create_flexible_obs_space, create_reward_space, DictEnv, LoggingEnv, PytorchEnv, RewardSpaceWrapper, VecOneEnv
from c4_gym.c4_env import C4Env
from agent.c4_model import C4Model
from agent.c4_player import C4Player
from lib.data_helper import get_game_data_filenames, write_game_data_to_file
from lib.model_helper import load_best_model_weight, save_as_best_model

logger = getLogger(__name__)


def start(config: SimpleNamespace):
    return SelfPlayWorker(config).start()


# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    """
    Worker which trains a chess model using self play data. ALl it does is do self play and then write the
    game data to file, to be trained on by the optimize worker.

    Attributes:
        :ivar Config config: config to use to configure this worker
        :ivar ChessModel current_model: model to use for self play
        :ivar Manager m: the manager to use to coordinate between other workers
        :ivar list(Connection) cur_pipes: pipes to send observations to and get back mode predictions.
        :ivar list((str,list(float))): list of all the moves. Each tuple has the observation in FEN format and
            then the list of prior probabilities for each action, given by the visit count of each of the states
            reached by the action (actions indexed according to how they are ordered in the uci move list).
    """
    def __init__(self, config: SimpleNamespace):
        self.flags = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list(
            [self.current_model.get_pipes(self.flags.search_threads) for _ in range(self.flags.max_processes)]
        )
        self.buffer = []

    def start(self):
        """
        Do self play and write the data to the appropriate file.
        """
        self.buffer = []

        futures = deque()
        with ProcessPoolExecutor(max_workers=self.flags.max_processes) as executor:
            for game_idx in range(self.flags.max_processes * 2):
                futures.append(executor.submit(self_play_buffer, self.flags, cur=self.cur_pipes))
            game_idx = 0
            while True:
                game_idx += 1
                start_time = time()
                env, data = futures.popleft().result()
                print(f"game {game_idx:3} duration={time() - start_time:4.1f}s "
                      f"winner={env.winner:2}\n{env.game_state.board}\n")

                self.buffer += data
                if (game_idx % self.flags.nb_game_in_file) == 0:
                    self.flush_buffer()
                    save_as_best_model(self.current_model, idx=game_idx)
                futures.append(executor.submit(self_play_buffer, self.flags, cur=self.cur_pipes)) # Keep it going

    def load_model(self):
        """
        Load the current best model
        :return ChessModel: current best model
        """
        model = C4Model(self.flags, is_actor=True)
        # if self.flags.opts.new or not load_best_model_weight(model):
        #     model.build()
        #     save_as_best_model(model)
        return model

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(os.getcwd(), self.flags.play_data_dir, self.flags.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, self.buffer))
        thread.start()
        self.buffer = []

    def remove_play_data(self):
        """
        Delete the play data from disk
        """
        files = get_game_data_filenames(self.flags.resource)
        if len(files) < self.flags.play_data.max_file_num:
            return
        for i in range(len(files) - self.flags.play_data.max_file_num):
            os.remove(files[i])


def self_play_buffer(flags, cur) -> (C4Env, list):
    """
    Play one game and add the play data to the buffer
    :param Config flags: config for how to play
    :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
        predictions. One will be removed from this list during the game, then added back
    :return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
        of data to be appended to the SelfPlayWorker.buffer
    """
    pipes = cur.pop() # borrow
    env = C4Env(
        flags=flags,
        act_space=flags.act_space(),
        obs_space=create_flexible_obs_space(flags),
        autoplay=True
    )
    reward_space = create_reward_space(flags)
    env = RewardSpaceWrapper(env, reward_space)
    env = env.obs_space.wrap_env(env)
    env = LoggingEnv(env, reward_space)
    env = VecOneEnv(env)
    env = PytorchEnv(env, flags.device)
    env = DictEnv(env)

    output = env.reset()

    p1 = C4Player(flags, pipes=pipes)
    p2 = C4Player(flags, pipes=pipes)

    while not output['done']:
        if env.game_state.is_p1_turn:
            action = p1.action(env, output)
        else:
            action = p2.action(env, output)
        output = env.step(action)

    p1_reward, p2_reward = output['reward'].tolist()[0]

    p1.finish_game(p1_reward)
    p2.finish_game(p2_reward)

    data = []
    for i in range(len(p1.moves)):
        data.append(p1.moves[i])
        if i < len(p2.moves):
            data.append(p2.moves[i])

    cur.append(pipes)
    return env, data