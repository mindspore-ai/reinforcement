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
"""GymMultiEnvironment class."""

import gym
from gym import spaces
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P


class GymMultiEnvironment(nn.Cell):
    """
    The GymMultiEnvironment class provides the functions to interact with
    different environments. It is the multi-environment version of GymEnvironment.

    Args:
        params (dict): A dictionary contains all the parameters which are used to create the
            instance of GymEnvironment, such as name of environment, number of environment.
            Since this environment is based on Gym, the name of environment should match
            with the name in Gym.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0', 'env_nums': 10}
        >>> environment = GymMultiEnvironment(env_params)
        >>> print(environment)
        GymMultiEnvironment<>
    """

    def __init__(self,
                 params):
        super(GymMultiEnvironment, self).__init__(auto_prefix=False)
        self.params = params
        self._name = params['name']
        self._nums = params['env_nums']
        self._envs = []
        for _ in range(self._nums):
            self._envs.append(gym.make(self._name))

        self.np_to_ms_dtype_output = {
            np.dtype(np.float32): ms.float32,
            np.dtype(np.float64): ms.float32,
            np.dtype(np.int64): ms.int32,
            np.dtype(np.int32): ms.int32
        }

        self.np_to_ms_dtype_input = {
            np.dtype(np.float32): ms.float32,
            np.dtype(np.float64): ms.float64,
            np.dtype(np.int64): ms.int64,
            np.dtype(np.int32): ms.int32
        }

        self.np_to_ms_suitable_np_dtype_output = {
            np.dtype(np.float32): np.float32,
            np.dtype(np.float64): np.float32,
            np.dtype(np.int64): np.int32,
            np.dtype(np.int32): np.int32
        }

        pyfunc_state_shape = self._envs[0].observation_space.shape
        pyfunc_action_shape = self._envs[0].action_space.shape

        self._pyfunc_state_dtype = self._envs[0].observation_space.dtype
        self._pyfunc_action_dtype = self._envs[0].action_space.dtype

        self._state_space_dim = pyfunc_state_shape[0]
        action_space = self._envs[0].action_space
        if isinstance(action_space, spaces.Discrete):
            self._action_space_dim = action_space.n
        elif isinstance(action_space, spaces.Box):
            self._action_space_dim = action_space.shape[0]

        self.input_action_dtype = self.np_to_ms_dtype_input[self._pyfunc_action_dtype]
        self.output_state_dtype = self.np_to_ms_dtype_output[self._pyfunc_state_dtype]

        self.step_ops = P.PyFunc(self._step,
                                 [self.input_action_dtype,],
                                 [(self._nums,) + pyfunc_action_shape],
                                 [self.output_state_dtype, ms.float32, ms.bool_],
                                 [(self._nums,) + pyfunc_state_shape, (self._nums, 1), (self._nums, 1)])
        self.reset_ops = P.PyFunc(self._reset, [], [],
                                  [self.output_state_dtype,],
                                  [(self._nums,) + pyfunc_state_shape])

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state of each environment.

        Returns:
            A list of tensor which states for all the initial states of each environment.

        """

        return self.reset_ops()[0]

    def step(self, action):
        """
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), a list of environment state after performing the action.
            - reward (Tensor), a list of reward after performing the action.
            - done (Tensor), whether the simulations of each environment finishes or not
        """
        action = action.astype(self.input_action_dtype)
        return self.step_ops(action)

    def clone(self):
        """
        Make a copy of the environment.

        Returns:
            env (object). A copy of the original environment object.
        """

        env = GymMultiEnvironment(self.params)
        return env

    @property
    def state_space_dim(self):
        """
        Get the state space dim of the environment.

        Returns:
            A tuple which states for the space dimension of state
        """

        return self._state_space_dim

    @property
    def action_space_dim(self):
        """
        Get the action space dim of the environment.

        Returns:
            A tuple which states for the space dimension of action
        """

        return self._action_space_dim

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A list of numpy array which states for the initial state of each environment.
        """

        self._done = False
        s0 = [(env.reset()).astype(self.np_to_ms_suitable_np_dtype_output[self._pyfunc_state_dtype])
              for env in self._envs]
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(List[numpy.dtype]): The action which is calculated by policy net.
            It could be List[int] or List[float] or other else, according to different environment.

        Returns:
            - s1 (List[numpy.array]), a list of environment state after performing the action.
            - r1 (List[numpy.array]), a list of reward after performing the action.
            - done (List[boolean]), whether the simulations of each environment finishes or not
        """
        s1, r1, done = [], [], []
        for i in range(self._nums):
            s, r, d, _ = self._envs[i].step(action[i])
            s1.append(s.astype(self.np_to_ms_suitable_np_dtype_output[self._pyfunc_state_dtype]))
            r1.append(np.array([r]).astype(np.float32))
            done.append(np.array([d]))
        s1 = np.stack(s1)
        r1 = np.stack(r1)
        done = np.stack(done)
        return s1, r1, done
