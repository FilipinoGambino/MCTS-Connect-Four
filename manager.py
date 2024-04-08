"""
Manages starting off each of the separate processes involved in ChessZero -
self play, training, and evaluation.
"""

import hydra
from logging import getLogger, disable
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import wandb

from api_key import WANDB_KEY
from worker.utils import flags_to_namespace

logger = getLogger(__name__)

CMD_LIST = ['self', 'opt', 'eval', 'sl', 'uci']


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    # flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    # flags.setdefault("obs_space_kwargs", {})
    # flags.setdefault("reward_space_kwargs", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 10.)
    flags.setdefault("num_learner_threads", 1)
    # flags.setdefault("use_teacher", False)
    # flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)

    # Model params


    # Reloading previous run params
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_value_warmup_batches", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)

    return OmegaConf.create(flags)


@hydra.main(config_path="conf", config_name="conv_debug", version_base=None)
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()

    #TODO add this back?
    # if Path("config.yaml").exists():
    #     new_flags = OmegaConf.load("config.yaml")
    #     flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logger.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    logger.info(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "outputs/config.yaml")
    flags = flags_to_namespace(OmegaConf.to_container(flags))
    if not flags.disable_wandb:
        wandb.init(
            config=flags,
            project=flags.project,
            entity=flags.entity,
            group=flags.group,
            name=flags.name,
        )

    # mp.set_sharing_strategy(flags.sharing_strategy)
    start(flags)

def start(flags):
    """
    Starts one of the processes based on command line arguments.

    :return : the worker class that was started
    """

    logger.info(f"Running: {flags.worker_type}")

    if flags.worker_type == 'self_play':
        from worker import self_play
        return self_play.start(flags)
    elif flags.worker_type == 'optimize':
        from worker import optimize
        return optimize.start(flags)
    elif flags.worker_type == 'evaluate':
        from worker import evaluate
        return evaluate.start(flags)
    # elif args.worker_type == 'sl':
    #     from worker import sl
    #     return sl.start(config)
    # elif args.worker_type == 'uci':
    #     from .play_game import uci
    #     return uci.start(config)


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    try:
        wandb.login(key=WANDB_KEY)
    except NameError:
        pass
    main()