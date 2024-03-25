import hydra
import logging
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import wandb

try:
    from api_key import WANDB_KEY
except ImportError:
    pass
from utils import flags_to_namespace
from mcts.mcts import train

os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("num_buffers", max(2 * flags["num_actors"], flags["batch_size"] // flags["n_actor_envs"]))
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})

    # Training params
    flags.setdefault("use_mixed_precision", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 10.)
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)

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


@hydra.main(config_path="conf", config_name="conv_phase1", version_base=None)
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()
    #TODO add this back?\/

    # if Path("config.yaml").exists():
    #     new_flags = OmegaConf.load("config.yaml")
    #     flags = OmegaConf.merge(new_flags, cli_conf)

    if flags.get("load_dir", None) and not flags.get("weights_only", False):
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        logging.info("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    logging.info(OmegaConf.to_yaml(flags, resolve=True))
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
    train(flags)


if __name__ == "__main__":
    # mp.set_start_method("spawn")
    try:
        wandb.login(key=WANDB_KEY)
    except NameError:
        pass
    main()