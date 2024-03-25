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