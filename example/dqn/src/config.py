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
DQN config.
"""

import mindspore as ms
from mindspore_rl.environment import GymEnvironment
from .dqn import DQNActor, DQNLearner, DQNPolicy

learner_params = {'gamma': 0.99}
trainer_params = {
    'num_evaluate_episode': 10,
    'ckpt_path': './ckpt',
}

env_params = {'name': 'CartPole-v0'}
eval_env_params = {'name': 'CartPole-v0'}

policy_params = {
    'epsi_high': 0.1,
    'epsi_low': 0.1,
    'decay': 200,
    'lr': 0.001,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size': 100,
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': DQNActor,
        'params': None,
        'policies': ['init_policy', 'collect_policy', 'evaluate_policy'],
        'networks': ['policy_network', 'target_network'],
        'environment': True,
        'eval_environment': True,
        'replay_buffer': {'capacity': 100000,
                          'shape': [(4,), (1,), (1,), (4,)],
                          'type': [ms.float32, ms.int32, ms.float32, ms.float32],
                          'sample_size': 64},
    },
    'learner': {
        'number': 1,
        'type': DQNLearner,
        'params': learner_params,
        'networks': ['target_network', 'policy_network_train']
    },
    'policy_and_network': {
        'type': DQNPolicy,
        'params': policy_params
    },
    'environment': {
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': GymEnvironment,
        'params': eval_env_params
    },
}
