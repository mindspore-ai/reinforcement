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
AC config.
"""

from mindspore_rl.environment import GymEnvironment
from .ac import ACPolicyAndNetwork, ACLearner, ACActor

env_params = {'name': 'CartPole-v0'}
policy_params = {
    'alr': 0.001,
    'clr': 0.01,
    'state_space_dim': 4,
    'action_space_dim': 2,
    'hidden_size': 20,
    'gamma': 0.9,
}
trainer_params = {
    'num_evaluate_episode': 10,
    'ckpt_path': './ckpt',
}
learner_params = {
    'gamma': 0.9,
    'state_space_dim': 4,
    'action_space_dim': 2,
}
algorithm_config = {
    'actor': {
        'number': 1,
        'type': ACActor,
        'params': None,
        'policies': [],
        'networks': ['actor_net'],
        'environment': True,
        'eval_environment': True,
    },
    'learner': {
        'number': 1,
        'type': ACLearner,
        'params': learner_params,
        'networks': ['actor_net_train', 'actor_net', 'critic_net_train', 'critic_net']
    },
    'policy_and_network': {
        'type': ACPolicyAndNetwork,
        'params': policy_params
    },
    'environment': {
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': GymEnvironment,
        'params': env_params
    },
}
