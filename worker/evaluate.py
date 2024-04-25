"""
Encapsulates the worker which evaluates newly-trained models and picks the best one
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from multiprocessing import Manager
from types import SimpleNamespace

from c4_gym import create_flexible_obs_space, create_reward_space, DictEnv, LoggingEnv, PytorchEnv, RewardSpaceWrapper, VecOneEnv
from agent.c4_model import C4Model
from agent.c4_player import C4Player
from c4_gym.c4_env import C4Env

logger = getLogger(__name__)


def start(flags: SimpleNamespace):
    return EvaluateWorker(flags).start()

class EvaluateWorker:
    """
    Worker which evaluates trained models and keeps track of the best one

    Attributes:
        :ivar Config config: config to use for evaluation
        :ivar PlayConfig config: PlayConfig to use to determine how to play, taken from config.eval.play_config
        :ivar C4Model current_model: currently chosen best model
        :ivar Manager m: multiprocessing manager
        :ivar list(Connection) cur_pipes: pipes on which the current best ChessModel is listening which will be used to
            make predictions while playing a game.
    """
    def __init__(self, flags: SimpleNamespace):
        """
        :param flags: Config to use to control how evaluation should work
        """
        self.flags = flags
        self.current_model = self.load_current_model()
        self.nextgen_model = self.load_next_generation_model()
        self.m = Manager()
        self.cur_pipes = self.m.list([self.current_model.get_pipes(self.flags.search_threads) for _ in range(self.flags.max_processes)])
        self.ng_pipes = self.m.list([self.nextgen_model.get_pipes(self.flags.search_threads) for _ in range(self.flags.max_processes)])

    def start(self):
        logger.info(f"Starting evaluation of the nextgen model")
        if self.evaluate_model():
            logger.info(f"The nextgen model is better")
            self.nextgen_model.save_as_best_model()

    def evaluate_model(self):
        """
        Given a model, evaluates it by playing a bunch of games against the current model.

        :return: true iff this model is better than the current_model
        """

        with ProcessPoolExecutor(max_workers=self.flags.max_processes) as executor:
            futures = [
                executor.submit(play_game, self.flags, self.cur_pipes, self.ng_pipes, (game_idx % 2 == 0))
                for game_idx in range(self.flags.eval_games)
            ]

            results = []
            for game in as_completed(futures):
                nextgen_score, env = game.result()
                results.append(nextgen_score)
                win_rate = sum(results) / len(results)
                game_idx = len(results)
                logger.info(f"Game {game_idx:>3}: Win_rate={win_rate*100:3.2f}%\n{env.board}\n")

        win_rate = sum(results) / len(results)
        logger.info(f"Winning rate {win_rate*100:.2f}%")
        return win_rate >= self.flags.replace_rate

    def load_current_model(self):
        """
        Loads the best model from the standard directory.
        :return C4Model: the last best model to evaluate against
        """
        model = C4Model(self.flags, self.flags.actor_device, self.flags.current_model_weight_fname)
        return model

    def load_next_generation_model(self):
        """
        Loads the next generation model from the standard directory
        :return C4Model: the next gen model to evaluate
        """
        model = C4Model(self.flags, self.flags.actor_device, self.flags.nextgen_model_weight_fname)
        return model


def play_game(flags, cur, ng, switch_p1: bool) -> (float, C4Env, bool):
    """
    Plays a game against models cur and ng and reports the results.

    :param SimpleNamespace flags: config for how to play the game
    :param ChessModel cur: should be the current model
    :param ChessModel ng: should be the next generation model
    :param bool switch_p1: whether cur should play white or black
    :return (float, ChessEnv, bool): the score for the ng model
        (0 for loss, .5 for draw, 1 for win), the env after the game is finished, and a bool
        which is true iff cur played as white in that game.
    """
    cur_pipes = cur.pop()
    ng_pipes = ng.pop()
    env = C4Env(
        flags=flags,
        act_space=flags.act_space(),
        obs_space=create_flexible_obs_space(flags),
    )
    reward_space = create_reward_space(flags)
    env = RewardSpaceWrapper(env, reward_space)
    env = env.obs_space.wrap_env(env)
    env = LoggingEnv(env, reward_space)
    env = VecOneEnv(env)
    env = PytorchEnv(env, flags.actor_device)
    env = DictEnv(env)

    env_output = env.reset()

    current_player = C4Player(flags, pipes=cur_pipes)
    nextgen_player = C4Player(flags, pipes=ng_pipes)

    if switch_p1:
        p1, p2 = nextgen_player, current_player
    else:
        p2, p1 = nextgen_player, current_player

    while not env.done:
        if env.game_state.is_p1_turn:
            action = p1.action(env, env_output)
        else:
            action = p2.action(env, env_output)
        env_output = env.step(action)

    winner = env.winner
    if not winner:
        nextgen_score = 0.5
    elif winner == switch_p1 or winner - 2 == switch_p1:
        nextgen_score = 1.
    else:
        nextgen_score = 0.

    cur.append(cur_pipes)
    ng.append(ng_pipes)
    return nextgen_score, env