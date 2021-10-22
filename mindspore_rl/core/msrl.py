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
Implementation of MSRL class.
"""

import inspect
from mindspore_rl.core import ReplayBuffer
from mindspore_rl.environment import GymMultiEnvironment
from mindspore_rl.agent import Agent
import mindspore.nn as nn
from mindspore.ops import operations as P


class MSRL(nn.Cell):
    """
    The MSRL class provides the function handlers and APIs for reinforcement
    learning algorithm development.

    It exposes the following function handler to the user. The input and output of
    these function handlers are identical to the user defined functions.

    .. code-block::

        agent_act_init
        agent_act_collect
        agent_act_eval
        agent_act
        agent_reset
        sample_buffer
        agent_learn

    Args:
        config(dict): provides the algorithm configuration.

            - Top level: defines the algorithm components.

              - key: 'actor',              value: the actor configuration (dict).
              - key: 'learner',            value: the learner configuration (dict).
              - key: 'policy_and_network', value: the policy and networks used by
                actors and learners (dict).
              - key: 'env',                value: the environment configuration (dict).

            - Second level: the configuration of each algorithm component.

              - key: 'number',      value: the number of actors/learner (int).
              - key: 'type',        value: the type of the
                actor/learner/policy_and_network/environment (class name).
              - key: 'params',      value: the parameters of
                actor/learner/policy_and_network/environment (dict).
              - key: 'policies',    value: the list of policies used by the
                actor/learner (list).
              - key: 'networks',    value: the list of networks used by the
                actor/learner (list).
              - key: 'environment', value: True if the component needs to interact
                with the environment, False otherwise (Bool).
              - key: 'buffer',      value: the buffer configuration (dict).
    """

    def __init__(self, config):
        super(MSRL, self).__init__()
        self.actors = []
        self.learner = None
        self.trainer = None
        self.policies = {}
        self.network = {}
        self.envs = []
        self.agent = None
        self.buffers = []
        self.env_num = 1

        # apis
        self.agent_act_init = None
        self.agent_act = None
        self.agent_evaluate = None
        self.update_buffer = None
        self.agent_reset = None
        self.sample_buffer = None
        self.agent_learn = None
        self.buffer_full = None

        for item in ['environment', 'policy_and_network', 'actor', 'learner']:
            if item not in config:
                raise ValueError(f"The `{item}` configuration should be provided.")

        self.init(config)

    def _create_instance(self, sub_config, actor_id=None):
        """
        Create class object from the configuration file, and return the instance of 'type' in
        input sub_config.

        Args:
            sub_config (dict): configuration file of the class.
            actor_id (int): the id of the actor. Default: None.

        Returns:
            obj (object), the class instance.
        """

        class_type = sub_config['type']
        params = sub_config['params']
        if actor_id is None:
            obj = class_type(params)
        else:
            obj = class_type(params, actor_id)
        return obj

    def __create_environments(self, config):
        """
        Create the environments object from the configuration file, and return the instance
        of environment and evaluation environment.

        Args:
            config (dict): algorithm configuration file.

        Returns:
            - env (object), created environment object.
            - eval_env (object), created evaluation environment object.
        """

        env = None
        if 'number' in config['environment']:
            self.env_num = config['environment']['number']

        if self.env_num > 1:
            config['environment']['type'] = GymMultiEnvironment
            config['environment']['params']['env_nums'] = self.env_num
            config['eval_environment']['type'] = GymMultiEnvironment
            config['eval_environment']['params']['env_nums'] = 1

        env = self._create_instance(config['environment'])

        if 'eval_environment' in config:
            eval_env = self._create_instance(config['eval_environment'])
        return env, eval_env

    def __params_generate(self, config, obj, target, attribute):
        """
        Parse the input object to generate parameters, then store the parameters into
        the dictionary of configuration

        Args:
            config (dict): the algorithm configuration.
            obj (object): the object for analysis.
            target (string): the name of the target class.
            attribute (string): the name of the attribute to parse.

        """

        for attr in inspect.getmembers(obj):
            if attr[0] in config[target][attribute]:
                config[target]['params'][attr[0]] = attr[1]

    def __create_replay_buffer(self, config):
        """
        Create the replay buffer object from the configuration file, and return the instance
        of replay buffer.

        Args:
            config (dict): the configuration for the replay buffer.

        Returns:
            replay_buffer (object), created replay buffer object.
        """

        capacity = config['capacity']
        buffer_shapes = config['shape']
        sample_size = config['sample_size']
        types = config['type']
        replay_buffer = ReplayBuffer(
            sample_size, capacity, buffer_shapes, types)
        return replay_buffer

    def init(self, config):
        """
        Initialization of MSRL object.
        The function creates all the data/objects that the algorithm requires.
        It also initializes all the function handler.

        Args:
            config (dict): algorithm configuration file.
        """

        env, eval_env = self.__create_environments(config)
        state_space_dim = env.state_space_dim

        if 'state_space_dim' in config['policy_and_network']['params']:
            config['policy_and_network']['params']['state_space_dim'] = state_space_dim
        # special cases
        if 'action_space_dim' in config['policy_and_network']['params'] and \
                config['policy_and_network']['params']['action_space_dim'] == 0:
            action_space_dim = env.action_space_dim
            config['policy_and_network']['params']['action_space_dim'] = action_space_dim

        policy_and_network = self._create_instance(
            config['policy_and_network'])

        if 'params' not in config['actor'] or config['actor']['params'] is None:
            config['actor']['params'] = {}
        if 'params' not in config['learner'] or config['learner']['params'] is None:
            config['learner']['params'] = {}

        if 'policies' in config['actor']:
            self.__params_generate(config, policy_and_network, 'actor', 'policies')

        if 'networks' in config['actor']:
            self.__params_generate(config, policy_and_network, 'actor', 'networks')

        if 'environment' in config['actor']:
            config['actor']['params']['environment'] = env
            config['actor']['params']['eval_environment'] = eval_env

        if 'replay_buffer' in config['actor']:  # doesn't work for multi buffer
            conf = config['actor']['replay_buffer']
            self.buffers = self.__create_replay_buffer(conf)
            config['actor']['params']['replay_buffer'] = self.buffers

        if 'networks' in config['learner']:
            self.__params_generate(config, policy_and_network, 'learner', 'networks')

        actor_num = config['actor']['number']

        if actor_num == 1:
            self.actors = self._create_instance(config['actor'])
        else:
            raise ValueError("Sorry, the current version only support one actor!")

        if 'state_space_dim' in config['learner']['params']:
            config['learner']['params']['state_space_dim'] = env.state_space_dim

        self.learner = self._create_instance(config['learner'])
        self.agent = Agent(actor_num, self.actors, self.learner)

        self.agent_act = self.actors.act
        self.agent_act_init = self.actors.act_init
        self.agent_evaluate = self.actors.evaluate
        self.agent_update = self.actors.update
        self.agent_reset_collect = self.actors.reset_collect_actor
        self.agent_reset_eval = self.actors.reset_eval_actor
        self.agent_learn = self.learner.learn

        if self.buffers:
            self.replay_buffer_sample = self.buffers.sample
            self.replay_buffer_insert = self.buffers.insert
            self.replay_buffer_full = self.buffers.full
            self.replay_buffer_reset = self.buffers.reset

    def get_replay_buffer(self):
        """
        It will return the instance of replay buffer.

        Returns:
            Buffers (object), The instance of relay buffer. If the buffer is None, the return
            value will be None.
        """

        return self.buffers

    def get_replay_buffer_elements(self, transpose=False, shape=None):
        """
        It will return all the elements in the replay buffer.
        Args:
            transpose (boolean): whether the output element needs to be transpose,
            if transpose is true, shape will also need to be filled. Default: False
            shape (Tuple[int]): the shape used in transpose. Default: None

        Returns:
            elements (List[Tensor]), A set of tensor contains all the elements in the replay buffer
        """

        transpose_op = P.Transpose()
        elements = ()
        for e in self.buffers.buffer:
            if transpose:
                e = transpose_op(e, shape)
                elements += (e,)
            else:
                elements += (e,)

        return elements
