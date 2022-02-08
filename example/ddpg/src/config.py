# Copyright 2022 Huawei Technologies Co., Ltd
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
DDPG config.
"""

import mindspore
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core.replay_buffer import ReplayBuffer
from .ddpg import DDPGActor, DDPGLearner, DDPGPolicy

env_params = {'name': 'HalfCheetah-v2'}
eval_env_params = {'name': 'HalfCheetah-v2'}

policy_params = {
    'epsilon': 0.2,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 400,
    'hidden_size2': 300,
}

learner_params = {
    'gamma': 0.995,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'actor_lr': 1e-4,
    'critic_lr': 1e-3,
    'update_factor': 0.05,
    'update_interval': 5
}

trainer_params = {
    'init_collect_size': 1000,
    'duration': 1000,
    'batch_size': 64,
    'ckpt_path': './ckpt',
    'num_eval_episode': 10,
}

actor_params = {
    "damping": 0.15,
    "stddev": 0.2
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': DDPGActor,
        'params': actor_params,
        'policies': [],
        'networks': ['actor_net'],
        'environment': True,
        'eval_environment': True,

    },
    'learner': {
        'number': 1,
        'type': DDPGLearner,
        'params': learner_params,
        'networks': ['actor_net', 'critic_net', 'ddpg_net_train']
    },
    'policy_and_network': {
        'type': DDPGPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'number': 1,
        'type': GymEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'number': 1,
        'type': ReplayBuffer,
        'capacity': 100000,
        # obs, actions, rewards, next_obs, done
        'data_shape': [(17,), (6,), (1,), (17,), (1,)],
        'data_type': [
            mindspore.float32, mindspore.float32, mindspore.float32, mindspore.float32, mindspore.bool_
        ],
        'sample_size': 64
    }
}
