from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
import os
from threading import Thread
from time import time
from types import SimpleNamespace

from c4_gym import create_env
from c4_gym.c4_env import C4Env
from agent.c4_model import C4Model
from agent.c4_player import C4Player
from lib.data_helper import write_game_data_to_file

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
        :ivar C4Model current_model: model to use for self play
        :ivar Manager m: the manager to use to coordinate between other workers
        :ivar list(Connection) cur_pipes: pipes to send observations to and get back mode predictions.
        :ivar list((str,list(float))): list of all the moves. Each tuple has the observation in FEN format and
            then the list of prior probabilities for each action, given by the visit count of each of the states
            reached by the action (actions indexed according to how they are ordered in the uci move list).
    """
    def __init__(self, config: SimpleNamespace):
        self.flags = config
        self.current_model = C4Model(self.flags, self.flags.actor_device, self.flags.current_model_weight_fname)
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

        with ProcessPoolExecutor(max_workers=self.flags.max_processes) as executor:
            futures = deque(
                executor.submit(self_play_buffer, self.flags, cur=self.cur_pipes)
                for _ in range(self.flags.max_processes * 2)
            )
            assert self.flags.self_play_games % self.flags.max_games_per_file == 0, "Excess info won't be saved"
            for game_idx in range(1, self.flags.self_play_games + 1):
                start_time = time()
                env, data = futures.popleft().result()
                print(f"Game {game_idx:3} Duration= {time() - start_time:2.1f}s "
                      f"Winner= {env.winner}\n{env.game_state.board}\n")

                self.buffer += data
                if (game_idx % self.flags.max_games_per_file) == 0:
                    self.flush_buffer()
                futures.append(executor.submit(self_play_buffer, self.flags, cur=self.cur_pipes))

    def flush_buffer(self):
        """
        Flush the play data buffer and write the data to the appropriate location
        """
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(os.getcwd(), self.flags.play_data_dir)
        fname = self.flags.play_data_filename_tmpl % game_id
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file, args=(path, fname, self.buffer))
        thread.start()
        self.buffer = []


def self_play_buffer(flags, cur) -> (C4Env, list):
    """
    Play one game and add the play data to the buffer
    :param Config flags: config for how to play
    :param list(Connection) cur: list of pipes to use to get a pipe to send observations to for getting
        predictions. One will be removed from this list during the game, then added back
    :return (ChessEnv,list((str,list(float)): a tuple containing the final ChessEnv state and then a list
        of data to be appended to the SelfPlayWorker.buffer
    """
    pipes = cur.pop()

    env = create_env(flags, flags.actor_device)

    env_output = env.reset()

    p1 = C4Player(flags, pipes=pipes)
    p2 = C4Player(flags, pipes=pipes)

    while not env.done:
        if env.game_state.is_p1_turn:
            action = p1.action(env, env_output)
        else:
            action = p2.action(env, env_output)

        env_output = env.step(action)

    p1_reward, p2_reward = env_output['reward'].tolist()[0]

    p1.finish_game(p1_reward)
    p2.finish_game(p2_reward)

    data = []
    for i in range(len(p1.moves)):
        data.append(p1.moves[i])
        if i < len(p2.moves):
            data.append(p2.moves[i])

    cur.append(pipes)
    return env, data