"""
Defines the process which will listen on the pipe for
an observation of the game state and return a prediction from the policy and
value network.
"""
from multiprocessing import connection, Pipe, Process
from threading import Thread
import torch
from logging import getLogger
from torch.nn.functional import softmax

logger = getLogger(__name__)


class C4API:
    """
    Defines the process which will listen on the pipe for
    an observation of the game state and return the predictions from the policy and
    value networks.
    Attributes:
        :ivar C4Model agent_model: C4Model to use to make predictions.
        :ivar list(Connection): list of pipe connections to listen for states on and return predictions on.
    """
    # noinspection PyUnusedLocal
    def __init__(self, agent_model):
        """

        :param C4Model agent_model: model used to make predictions
        """
        self.agent_model = agent_model
        self.pipes = []
        self.flags = agent_model.flags
        self.predictors = []

    def start(self):
        """
        Starts a thread to listen on the pipe and make predictions
        :return:
        """
        prediction_worker = Process(target=self._predict_batch_worker, name=f"prediction_worker", daemon=True)
        prediction_worker.start()

    def create_pipe(self):
        """
        Creates a new two-way pipe and returns the connection to one end of it (the other will be used
        by this class)
        :return Connection: the other end of this pipe.
        """
        me, you = Pipe()
        self.pipes.append(me)
        return you

    @torch.no_grad()
    def _predict_batch_worker(self):
        """
        Thread worker which listens on each pipe in self.pipes for an observation, and then outputs
        the predictions for the policy and value networks when the observations come in. Repeats.
        """
        while self.pipes:
            for pipe in connection.wait(self.pipes, timeout=0.1): # Returns pipes that are ready OR when the other side closes
                try:
                    obs = pipe.recv()

                    output = self.agent_model.model.sample_actions(obs)

                    policy_ary = softmax(output['policy_logits'][0], dim=0).to(dtype=torch.double)
                    action_mask = output['aam'].squeeze(dim=0)
                    policy_ary = torch.where(action_mask, float("-inf"), policy_ary).tolist()
                    value_ary = output['baseline'].item()

                    pipe.send((policy_ary, value_ary))
                except EOFError: # Triggers when the other side of the pipe is closed
                    for thread in self.predictors:
                        thread.join()
                    return