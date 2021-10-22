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
PPO training example.
"""

from ppo import PPOActor, PPOLearner, PPOPolicy
from ppo_trainer import PPOTrainer
from mindspore_rl.environment import GymEnvironment
from mindspore_rl.core import Session
import mindspore
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)

env_params = {'name': 'HalfCheetah-v2'}
eval_env_params = {'name': 'HalfCheetah-v2'}

policy_params = {
    'epsilon': 0.2,
    'lr': 1e-3,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 200,
    'hidden_size2': 100,
    'sigma_init_std': 0.35,
    'critic_coef': 0.5,
}

learner_params = {
    'gamma': 0.99,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'iter_times': 25
}

trainer_params = {
    'duration': 1000,
    'batch_size': 1,
    'eval_interval': 20,
    'num_eval_episode': 30,
    'metrics': False,
}

ppo_algorithm_config = {
    'actor': {
        'number': 1,
        'type': PPOActor,
        'params': None,
        'policies': [],
        'networks': ['actor_net'],
        'environment': True,
        'eval_environment': True,
        'replay_buffer': {
            'capacity':
            1000,
            'shape': [(30, 17), (30, 6), (30, 1), (30, 17), (30, 6), (30, 6)],
            'type': [
                mindspore.float32, mindspore.float32, mindspore.float32,
                mindspore.float32, mindspore.float32, mindspore.float32,
            ],
            'sample_size': 1,
        }
    },
    'learner': {
        'number': 1,
        'type': PPOLearner,
        'params': learner_params,
        'networks': ['critic_net', 'ppo_net_train']
    },
    'policy_and_network': {
        'type': PPOPolicy,
        'params': policy_params
    },
    'environment': {
        'number': 30,
        'type': GymEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'type': GymEnvironment,
        'params': eval_env_params
    }
}

ppo_session = Session(ppo_algorithm_config)
ppo_session.run(class_type=PPOTrainer, episode=1000, params=trainer_params)
