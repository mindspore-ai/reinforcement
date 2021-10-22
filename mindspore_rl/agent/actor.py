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
Implementation of Actor base class.
"""

import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np


class Actor(nn.Cell):
    r"""
    Base class for all actors.

    Examples:
        >>> from mindspore_rl.agent.actor import Actor
        >>> from mindspore_rl.network import FullyConnectedNet
        >>> from mindspore_rl.environment import GymEnvironment
        >>> class MyActor(Actor):
        ...   def __init__(self):
        ...     super(MyActor, self).__init__()
        ...     self.argmax = P.Argmax()
        ...     self.actor_net = FullyConnectedNet(4, 10, 2)
        ...     self.env = GymEnvironment({'name': 'CartPole-v0'})
        >>> my_actor = MyActor()
        >>> print(my_actor)
        MyActor<
        (actor_net): FullyConnectedNet<
        (linear1): Dense<input_channels=4, output_channels=10, has_bias=True>
        (linear2): Dense<input_channels=10, output_channels=2, has_bias=True>
        (relu): ReLU<>
        >
        (environment): GymEnvironment<>
    """

    def __init__(self):
        super(Actor, self).__init__(auto_prefix=False)
        self._environment = None
        self._eval_env = None
        self.false = Tensor(np.array([False,]), ms.bool_)

    def act(self, state):
        """
        The interface of the act function.
        User will need to overload this function according to
        the algorithm. But argument of this function should be
        the state output from the environment.

        Args:
            state (Tensor): the output state from the environment.

        Returns:
            - done (Tensor), whether the simulation is finish or not.
            - reward (Tensor), simulation reward.
            - state (Tensor), simulation state.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def act_init(self, state):
        """
        The interface of the act initialization function.
        User will need to overload this function according to
        the algorithm. But argument of this function should be
        the state output from the environment.

        Args:
            state (Tensor): the output state from the environment.

        Returns:
            - done (Tensor), whether the simulation is finish or not.
            - reward (Tensor), simulation reward.
            - state (Tensor), simulation state.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def evaluate(self, state):
        """
        The interface of the act evaluation function.
        User will need to overload this function according to
        the algorithm. But argument of this function should be
        the state output from the environment.

        Args:
            state (Tensor): the output state from the environment.

        Returns:
            - done (Tensor), whether the simulation is finish or not.
            - reward (Tensor), simulation reward.
            - state (Tensor), simulation state.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def update(self):
        """
        The interface of the update function.
        User will need to overload this function according to the algorithm.
        """

        raise NotImplementedError("Method should be overridden by subclass.")

    def env_setter(self, env):
        """
        Set the environment by the input `env` for the actor. The `env` is created by
        class `GymEnvironment` or other environment class.

        Args:
            env (object): the input environment.

        Returns:
            environment.
        """

        self._environment = env
        return self._environment

    def reset_collect_actor(self):
        """
        Reset the collect actor, reset the collect actor's environment and
        return the reset state and a false flag of `done`.

        Returns:
            - state (Tensor), the state of the actor after reset.
            - Tensor, always false of `done`.
        """

        state = self._environment.reset()
        return state, self.false

    def reset_eval_actor(self):
        """
        Reset the eval actor, reset the eval actor's environment and
        return the reset state and a false flag of `done`.

        Return:
            - state (Tensor), the state of the actor after reset.
            - Tensor, always false of `done`.
        """

        state = self._eval_env.reset()
        return state, self.false
