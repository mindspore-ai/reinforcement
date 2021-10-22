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
The GymEnvironment base class.
"""

import gym
from gym import spaces
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P


class GymEnvironment(nn.Cell):
    """
    The GymEnvironment class provides the functions to interact with
    different environments.

    Args:
        params (dict): A dictionary contains all the parameters which are used to create the
            instance of GymEnvironment, such as the name of environment. Since this environment
            is based on Gym, the name of environment should match with the name in Gym.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0'}
        >>> environment = GymEnvironment(env_params)
        >>> print(environment)
        GymEnvironment<>
    """

    def __init__(self,
                 params):
        super(GymEnvironment, self).__init__(auto_prefix=False)
        self.params = params
        self._name = params['name']
        self._env = gym.make(self._name)

        np_to_ms_dtype_output = {
            np.dtype(np.float32): ms.float32,
            np.dtype(np.float64): ms.float32,
            np.dtype(np.int64): ms.int32,
            np.dtype(np.int32): ms.int32
        }

        np_to_ms_dtype_input = {
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

        pyfunc_state_shape = self._env.observation_space.shape
        pyfunc_action_shape = self._env.action_space.shape

        self._pyfunc_state_dtype = self._env.observation_space.dtype
        self._pyfunc_action_dtype = self._env.action_space.dtype

        self._state_space_dim = pyfunc_state_shape[0]
        action_space = self._env.action_space
        if isinstance(action_space, spaces.Discrete):
            self._action_space_dim = action_space.n
        elif isinstance(action_space, spaces.Box):
            self._action_space_dim = action_space.shape[0]

        self.input_action_dtype = np_to_ms_dtype_input[self._pyfunc_action_dtype]

        # step op
        step_input_type = [self.input_action_dtype,]
        step_input_shape = [pyfunc_action_shape,]
        step_output_type = [
            np_to_ms_dtype_output[self._pyfunc_state_dtype], ms.float32, ms.bool_]
        step_output_shape = [pyfunc_state_shape, (1,), (1,)]
        self.step_ops = P.PyFunc(
            self._step, step_input_type, step_input_shape, step_output_type, step_output_shape)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [np_to_ms_dtype_output[self._pyfunc_state_dtype],]
        reset_output_shape = [pyfunc_state_shape,]
        self.reset_ops = P.PyFunc(self._reset, reset_input_type,
                                  reset_input_shape, reset_output_type, reset_output_shape)

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """

        return self.reset_ops()[0]

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (mindspore.bool\_), whether the simulation finishes or not.
        """
        action = action.astype(self.input_action_dtype)
        return self.step_ops(action)

    def clone(self):
        """
        Make a copy of the environment.

        Returns:
            env (object), a copy of the original environment object.
        """

        env = GymEnvironment(self.params)
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
            A numpy array which states for the initial state of environment.
        """

        self._done = False
        s0 = self._env.reset()
        s0 = s0.astype(
            self.np_to_ms_suitable_np_dtype_output[self._pyfunc_state_dtype])
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(int or float): The action which is calculated by policy net. It could be integer
            or float, according to different environment

        Returns:
            - s1 (numpy.array), the environment state after performing the action.
            - r1 (numpy.array), the reward after performing the action.
            - done (boolean), whether the simulation finishes or not.
        """

        s1, r1, done, _ = self._env.step(action)
        s1 = s1.astype(
            self.np_to_ms_suitable_np_dtype_output[self._pyfunc_state_dtype])
        r1 = np.array([r1]).astype(np.float32)
        done = np.array([done])
        return s1, r1, done
