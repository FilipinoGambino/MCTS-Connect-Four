import logging
import math
from omegaconf import OmegaConf
from pathlib import Path
import pprint
import os
import threading
import time
import timeit
import torch
from torch.cuda import amp
from torch import nn
import traceback
from types import SimpleNamespace
from typing import Dict, Optional, Union
import multiprocessing as mp
import wandb

from c4_game.mcts_game import MCTSGame
from c4_gym import create_env
from nodes.mcts_node import Node
from nns import create_model
from utils import flags_to_namespace
from core.buffer_utils import Buffers, buffers_apply, create_buffers, fill_buffers_inplace, stack_buffers, split_buffers
from core import prof

# https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

class ClassThread(threading.Thread):
    def __init__(self, target, name, args):
        super().__init__()
        self.target = target
        self.name = name
        self.args = args

    def run(self):
        try:
            print('running thread')
            self.target(self.args)
        except Exception as e:
            print(f'An exception occured in thread "{self.name}":\n{e}')
            return

def mcts_self_play(flags, mcts_tree, actor_model):
    game_state = MCTSGame

    # Establish initial move probabilities
    _, probs, val = actor_model(game_state)
    mcts_tree.update_probs(probs)

    leaves = [mcts_tree.get_leaf() for _ in range(flags.n_actor_envs)]
    actions = torch.tensor([leaf.action for leaf in leaves]).unsqueeze(-1)
    env_output = env.step(actions)
    _, probs, vals = actor_model(env_output)
    for prob, val, leaf in zip(probs, val, leaves):
        leaf.backward(prob, val)

def act(
        flags,
        teacher_flags: Optional[SimpleNamespace],
        actor_index,
        free_queue,
        full_queue,
        actor_model,
        mcts_tree,
        buffers
):
    if flags.debug:
        catch_me = AssertionError
    else:
        catch_me = Exception
    try:
        logging.info(f"Actor {actor_index} started.")

        timings = prof.Timings()

        env = create_env(flags, torch.device("cpu"), teacher_flags=teacher_flags)
        env_output = env.reset()

        agent_output = mcts_tree.get_next_action()

        while True:
            index = free_queue.get()
            if index is None:
                break
            fill_buffers_inplace(buffers[index], dict(**env_output, **agent_output), 0)

            # Do new rollout.
            for t in range(flags.unroll_length):
                leaf = mcts_tree.get_next_action()
                env.step(leaf.action)
                probs, value = actor_model(output['obs'])
                leaf.backward(probs, value)

                full_queue.put(index)

        if actor_index == 0:
            logging.info(f"Actor {actor_index}: {timings.summary()}")

    except KeyboardInterrupt:
        pass  # Return silently.
    except catch_me as e:
        logging.error(f"Exception in worker process {actor_index}")
        traceback.print_exc()
        print()
        raise e

def learn(
        flags: SimpleNamespace,
        actor_model: nn.Module,
        learner_model: nn.Module,
        teacher_model: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        grad_scaler: amp.grad_scaler,
        lr_scheduler: torch.optim.lr_scheduler,
        total_games_played: int,
        baseline_only: bool = False,
        lock=threading.Lock(),
):
    return None, None

def get_batch(
    flags: SimpleNamespace,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    timings: prof.Timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(max(flags.batch_size // flags.n_actor_envs, 1))]
        timings.time("dequeue")
    batch = stack_buffers([buffers[m] for m in indices], dim=1)
    timings.time("batch")
    batch = buffers_apply(batch, lambda x: x.to(device=flags.learner_device, non_blocking=True))
    timings.time("device")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    return batch

def train(flags):
    # Necessary for multithreading and multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"

    if flags.num_buffers < flags.num_actors:
        raise ValueError("num_buffers should >= num_actors")
    if flags.num_buffers < flags.batch_size // flags.n_actor_envs:
        raise ValueError("num_buffers should be larger than batch_size // n_actor_envs")

    t = flags.unroll_length
    b = flags.batch_size

    if flags.use_teacher:
        teacher_flags = OmegaConf.load(Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))
    else:
        teacher_flags = None

    example_env = create_env(flags, torch.device("cpu"), teacher_flags=teacher_flags)
    buffers = create_buffers(
        flags,
        example_env.unwrapped[0].obs_space,
        example_env.reset(force=True)["info"]
    )
    del example_env

    if flags.load_dir:
        checkpoint_state = torch.load(Path(flags.load_dir) / flags.checkpoint_file, map_location=torch.device("cpu"))
    else:
        checkpoint_state = None

    actor_model = create_model(flags, flags.actor_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    actor_model.eval()
    actor_model.share_memory()

    mcts_tree = Node(flags, action=None)

    free_queue = mp.SimpleQueue()
    full_queue = mp.SimpleQueue()

    actor_processes = []
    for i in range(flags.num_actors):
        actor = ClassThread(
            target=mcts_self_play,
            name=f"actor_{i}",
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                actor_model,
                mcts_tree,
                buffers
            )
        )
        actor.start()
        actor_processes.append(actor)
        time.sleep(0.5)

    learner_model = create_model(flags, flags.learner_device, teacher_model_flags=teacher_flags, is_teacher_model=False)
    if checkpoint_state is not None:
        learner_model.load_state_dict(checkpoint_state["model_state_dict"])
    learner_model.train()
    learner_model = learner_model.share_memory()
    if not flags.disable_wandb:
        wandb.watch(learner_model, flags.model_log_freq, log="all", log_graph=True)

    def batch_and_learn(learner_idx, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, total_games_played, stats
        timings = prof.Timings()
        while step < flags.total_steps:
            timings.reset()
            full_batch = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                timings,
            )
            if flags.batch_size < flags.n_actor_envs:
                batches = split_buffers(full_batch, flags.batch_size, dim=1, contiguous=True)
            else:
                batches = [full_batch]
            for batch in batches:
                stats, total_games_played = learn(
                    flags=flags,
                    actor_model=actor_model,
                    learner_model=learner_model,
                    teacher_model=teacher_model,
                    batch=batch,
                    optimizer=optimizer,
                    grad_scaler=grad_scaler,
                    lr_scheduler=scheduler,
                    total_games_played=total_games_played,
                    baseline_only=step / (t * b) < flags.n_value_warmup_batches,
                )
                with lock:
                    step += t * b
                    if not flags.disable_wandb:
                        wandb.log(stats, step=step)
            timings.time("learn")
        if learner_idx == 0:
            logging.info(f"Batch and learn timing statistics: {timings.summary()}")

    # Load teacher model for KL loss
    if flags.use_teacher:
        if flags.teacher_kl_cost <= 0. and flags.teacher_baseline_cost <= 0.:
            raise ValueError("It does not make sense to use teacher when teacher_kl_cost <= 0 "
                             "and teacher_baseline_cost <= 0")
        teacher_model = create_model(
            flags,
            flags.learner_device,
            teacher_model_flags=teacher_flags,
            is_teacher_model=True
        )
        teacher_model.load_state_dict(
            torch.load(
                Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file,
                map_location=torch.device("cpu")
            )["model_state_dict"]
        )
        teacher_model.eval()
    else:
        teacher_model = None
        if flags.teacher_kl_cost > 0.:
            logging.warning(f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_kl_cost to 0.")
        if flags.teacher_baseline_cost > 0.:
            logging.warning(f"flags.teacher_baseline_cost is {flags.teacher_baseline_cost}, but use_teacher is False. "
                            f"Setting flags.teacher_baseline_cost to 0.")
        flags.teacher_kl_cost = 0.
        flags.teacher_baseline_cost = 0.

    def lr_lambda(epoch):
        min_pct = flags.min_lr_mod
        pct_complete = min(epoch * t * b, flags.total_steps) / flags.total_steps
        scaled_pct_complete = pct_complete * (1. - min_pct)
        return 1. - scaled_pct_complete

    optimizer = flags.optimizer_class(
        learner_model.parameters(),
        **flags.optimizer_kwargs
    )
    grad_scaler = amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    learner_threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,)
        )
        thread.start()
        # thread = MyThread(target=batch_and_learn, name=f"batch-and-learn-{i}", args=(i,))
        # thread.start()
        learner_threads.append(thread)


    def checkpoint(checkpoint_path: Union[str, Path]):
        logging.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
                "total_games_played": total_games_played,
            },
            checkpoint_path + ".pt",
        )
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
            },
            checkpoint_path + "_weights.pt"
        )

    for n in range(flags.num_buffers):
        free_queue.put(n)

    step, total_games_played, stats = 0, 0, {}

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(10)

            # Save every checkpoint_freq minutes
            if timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
                cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            bps = (step - start_step) / (t * b) / (timer() - start_time)
            logging.info(f"Steps {step:d} @ {sps:.1f} SPS / {bps:.1f} BPS. Stats:\n{pprint.pformat(stats)}")
    except KeyboardInterrupt:
        # Try checkpointing and joining actors then quit.
        return
    else:
        for thread in learner_threads:
            thread.join()
        logging.info(f"Learning finished after {step:d} steps.")
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)
        cp_path = str(step).zfill(int(math.log10(flags.total_steps)) + 1)
        checkpoint(cp_path)