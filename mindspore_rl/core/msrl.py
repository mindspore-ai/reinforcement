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

import mindspore.nn as nn
from mindspore.ops import operations as P

from mindspore_rl.environment import GymMultiEnvironment
from mindspore_rl.core.replay_buffer import ReplayBuffer


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
        self.collect_environment = None
        self.eval_environment = None

        # apis
        self.agent_act_init = None
        self.agent_act = None
        self.agent_evaluate = None
        self.update_buffer = None
        self.agent_reset = None
        self.sample_buffer = None
        self.agent_learn = None
        self.buffer_full = None

        compulsory_items = [
            'eval_environment', 'environment', 'policy_and_network',
            'actor', 'learner'
        ]
        self._compulsory_items_check(config, compulsory_items, 'config')

        self.init(config)

    def _compulsory_items_check(self, config, compulsory_item, position):
        for item in compulsory_item:
            if item not in config:
                raise ValueError(
                    f"The `{item}` configuration in `{position}` should be provided."
                )

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
        of environment and evaluate environment.

        Args:
            config (dict): algorithm configuration file.

        Returns:
            - env (object), created environment object.
            - eval_env (object), created evaluate environment object.
        """

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
        replay_buffer_config = config['actor']['replay_buffer']
        compulsory_item = ['capacity', 'shape', 'type']
        self._compulsory_items_check(replay_buffer_config, compulsory_item,
                                     'replay_buffer')

        num_replay_buffer = replay_buffer_config.get('number')
        capacity = replay_buffer_config['capacity']
        buffer_data_shapes = replay_buffer_config['shape']
        buffer_data_type = replay_buffer_config['type']

        sample_size = replay_buffer_config.get('sample_size')
        if not sample_size:
            sample_size = 32

        if (not num_replay_buffer) or num_replay_buffer == 1:
            buffer = ReplayBuffer(sample_size, capacity,
                                  buffer_data_shapes, buffer_data_type)
        else:
            buffer = [
                ReplayBuffer(sample_size, capacity, buffer_data_shapes,
                             buffer_data_type) for _ in num_replay_buffer
            ]
        return buffer

    def __create_policy_and_network(self, config):
        """
        Create an instance of XXXPolicy class in algorithm, it contains the networks. collect policy
        and eval policy of algorithm.

        Args:
            config (dict): A dictionary of configuration

        Returns:
            policy_and_network (object): The instance of policy and network
        """
        policy_and_network_config = config['policy_and_network']
        compulsory_items = ['type']
        self._compulsory_items_check(policy_and_network_config,
                                     compulsory_items, 'policy_and_network')

        params = policy_and_network_config.get('params')
        if params:
            if not params.get('state_space_dim'):
                config['policy_and_network']['params'][
                    'state_space_dim'] = self.collect_environment.observation_space.shape[-1]
            if not params.get('action_space_dim'):
                config['policy_and_network']['params'][
                    'action_space_dim'] = self.collect_environment.action_space.num_values

        policy_and_network = self._create_instance(policy_and_network_config)
        return policy_and_network

    def __create_actor(self, config, policy_and_network):
        """
        Create an instance of actor or a list of instances of actor

        Args:
            config (dict): A dictionary of configuration
            policy_and_network (object): The instance of policy_and_network

        Returns:
            actor (object or List(object)): An instance of actor a list of instances of actor
        """
        compulsory_items = ['number', 'type', 'policies']
        self._compulsory_items_check(config['actor'], compulsory_items,
                                     'actor')

        params = config['actor'].get('params')
        if not params:
            config['actor']['params'] = {}
        if config['actor'].get('environment'):
            config['actor']['params'][
                'environment'] = self.collect_environment
            config['actor']['params'][
                'eval_environment'] = self.eval_environment

        config['actor']['params']['replay_buffer'] = self.buffers

        if config['actor'].get('policies'):
            self.__params_generate(config, policy_and_network, 'actor',
                                   'policies')
        if config['actor'].get('networks'):
            self.__params_generate(config, policy_and_network, 'actor',
                                   'networks')

        num_actors = config['actor']['number']
        if num_actors == 1:
            actor = self._create_instance(config['actor'])
        else:
            raise ValueError(
                "Sorry, the current version only supports one actor !")
        return actor

    def __create_learner(self, config, policy_and_network):
        """
        Create an instance of learner or a list of instances of learner

        Args:
            config (dict): A dictionary of configuration
            policy_and_network (object): The instance of policy_and_network

        Returns:
            actor (object or List(object)): An instance of learner a list of instances of learner
        """
        compulsory_items = ['type', 'networks']
        self._compulsory_items_check(config['learner'], compulsory_items,
                                     'learner')

        params = config['actor'].get('params')
        if not params:
            config['learner']['params'] = {}
        if config['learner'].get('networks'):
            self.__params_generate(config, policy_and_network, 'learner',
                                   'networks')

        num_learner = config['learner']['number']
        if num_learner == 1:
            learner = self._create_instance(config['learner'])
        else:
            raise ValueError(
                "Sorry, the current version only supports one learner !")
        return learner

    def init(self, config):
        """
        Initialization of MSRL object.
        The function creates all the data/objects that the algorithm requires.
        It also initializes all the function handler.

        Args:
            config (dict): algorithm configuration file.
        """
        # ---------------------- Environment ----------------------
        self.collect_environment, self.eval_environment = self.__create_environments(
            config)

        # ---------------------- ReplayBuffer ----------------------
        replay_buffer = config['actor'].get('replay_buffer')
        if replay_buffer:
            self.buffers = self.__create_replay_buffer(config)
            self.replay_buffer_sample = self.buffers.sample
            self.replay_buffer_insert = self.buffers.insert
            self.replay_buffer_full = self.buffers.full
            self.replay_buffer_reset = self.buffers.reset

        # ---------------------- Agent ----------------------
        agent_config = config.get('agent')
        if not agent_config:
            policy_and_network = self.__create_policy_and_network(config)
            self.actors = self.__create_actor(config, policy_and_network)
            self.learner = self.__create_learner(config, policy_and_network)
            self.agent_act = self.actors.act
            self.agent_learn = self.learner.learn
        else:
            raise ValueError(
                "Sorry, the current does not support multi-agent yet")

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
