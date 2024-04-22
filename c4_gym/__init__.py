import torch

from . import act_spaces, obs_spaces, reward_spaces
from .c4_env import C4Env
from .wrappers import DictEnv, LoggingEnv, PytorchEnv, RewardSpaceWrapper, VecOneEnv
from . import act_spaces, obs_spaces, reward_spaces

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}

def create_env(flags, device: torch.device) -> PytorchEnv:
    env = C4Env(
        flags=flags,
        act_space=flags.act_space(),
        obs_space=create_obs_space(flags),
        autoplay=True
    )
    reward_space = create_reward_space(flags)
    env = RewardSpaceWrapper(env, reward_space)
    env = env.obs_space.wrap_env(env)
    env = LoggingEnv(env, reward_space)
    env = VecOneEnv(env)
    env = PytorchEnv(env, device)
    env = DictEnv(env)
    return env

def create_reward_space(flags) -> reward_spaces.BaseRewardSpace:
    return flags.reward_space(**flags.reward_space_kwargs)

def create_obs_space(flags) -> obs_spaces.BaseObsSpace:
    return flags.obs_space(**flags.obs_space_kwargs)