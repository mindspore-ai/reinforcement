# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Implementation of Agent base class.
"""

import mindspore.nn as nn


class Agent(nn.Cell):
    r"""
    The base class for the Agent.

    Args:
        num_actor(int): The actor numbers in this agent.
        actors(object): The actor instance.
        learner(object): The learner instance.

    Examples:
        >>> from mindspore_rl.agent.learner import Learner
        >>> from mindspore_rl.agent.actor import Actor
        >>> from mindspore_rl.agent.agent import Agent
        >>> actor_num = 1
        >>> actors = Actor()
        >>> learner = Learner()
        >>> agent = Agent(actor_num, actors, learner)
        >>> print(agent)
        Agent<
        (_actors): Actor<>
        (_learner): Learner<>
        >
    """

    def __init__(self, num_actor, actors, learner):
        super(Agent, self).__init__(auto_prefix=False)
        self._actors = actors
        self._num_actor = num_actor
        self._learner = learner

    def init(self):
        """
        Initialize the agent, reset all the actors in agent.
        """

        self.reset_all()

    def reset_all(self):
        """
        Reset the all the actors in agent, and return the reset `state`
        and the flag `done`.

        Returns:
            - state (Tensor), the state of the reset environment in actor.
            - done (Tensor), a false flag of `done`.
        """

        state, done = self._actors.reset()
        return state, done

    def act(self):
        """
        The act function interface.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def learn(self, samples):
        """
        The learn function interface.

        Args:
            samples (Tensor): the sample from replay buffer.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def update(self):
        """
        The update function interface.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def env_setter(self, env):
        """
        Set the agent environment for actors in agent.

        Args:
            env (object): the input environment.
        """
        self._actors.env_setter(env)

    @property
    def actors(self):
        """
        Get the instance of actors in the agent.

        Returns:
            actors (object), actors object created by class `Actor`.
        """
        return self._actors

    @property
    def num_actor(self):
        """
        Get the number of the actors of the agent.

        Returns:
            num_actor (int), actor numbers.
        """
        return self._num_actor

    @property
    def learner(self):
        """
        Get the instance of learner in the agent.

        Returns:
            learner (object), learner object created by class `Learner`.
        """
        return self._learner
