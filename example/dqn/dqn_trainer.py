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
"""DQN Trainer"""
import matplotlib.pyplot as plt

import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer


class DQNTrainer(Trainer):
    """DQN Trainer"""

    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.zero = Tensor(0, ms.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.zero_value = Tensor(0, ms.float32)
        self.fill_value = Tensor(1000, ms.float32)
        self.mod = P.Mod()
        self.all_ep_r = []
        self.all_steps = []
        self.evaluation_interval = params['evaluation_interval']
        self.num_evaluation_episode = params['num_evaluation_episode']
        self.metrics = params['metrics']
        self.update_period = Tensor(5, ms.float32)
        super(DQNTrainer, self).__init__(msrl)

    def train(self, episode):
        """Train DQN"""
        self.init_training()
        steps = 0
        for i in range(episode):
            if i % self.evaluation_interval == 0:
                reward = self.evaluation()
                reward = reward.asnumpy()
                print("-----------------------------------------")
                print(f"Evaluation result in episode {i} is {reward:.3f}")
                print("-----------------------------------------")
                if self.metrics:
                    self.all_steps.append(steps)
                    self.all_ep_r.append(reward)

            reward, episode_steps = self.train_one_episode(self.update_period)
            steps += episode_steps.asnumpy()
            print(f"Episode {i}, steps: {steps}, reward: {reward.asnumpy():.3f}")
        reward = self.evaluation()
        reward = reward.asnumpy()
        print("-----------------------------------------")
        print(f"Evaluation result in episode {i} is {reward:.3f}")
        print("-----------------------------------------")
        if self.metrics:
            self.all_ep_r.append(reward)
            self.all_steps.append(steps)
            self.plot()

    def plot(self):
        plt.plot(self.all_steps, self.all_ep_r)
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.savefig('dqn_rewards.png')

    @ms_function
    def init_training(self):
        """Initialize training"""
        state, done = self.msrl.agent_reset_collect()
        i = self.zero_value
        while self.less(i, self.fill_value):
            done, _, new_state, action, my_reward = self.msrl.agent_act_init(state)
            self.msrl.replay_buffer_insert([state, action, my_reward, new_state])
            state = new_state
            if done:
                state, done = self.msrl.agent_reset_collect()
            i += 1
        return done

    @ms_function
    def train_one_episode(self, update_period=5):
        """Train one episode"""
        state, done = self.msrl.agent_reset_collect()
        total_reward = self.zero
        steps = self.zero
        while not done:
            done, r, new_state, action, my_reward = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert([state, action, my_reward, new_state])
            state = new_state
            r = self.squeeze(r)
            self.msrl.agent_learn(self.msrl.replay_buffer_sample())
            total_reward += r
            steps += 1
            if not self.mod(steps, update_period):
                self.msrl.agent_update()
        return total_reward, steps

    @ms_function
    def evaluation(self):
        """Policy evaluation"""
        total_reward = self.zero_value
        for _ in msnp.arange(self.num_evaluation_episode):
            episode_reward = self.zero_value
            state, done = self.msrl.agent_reset_eval()
            while not done:
                done, r, state = self.msrl.agent_evaluate(state)
                r = self.squeeze(r)
                episode_reward += r
            total_reward += episode_reward
        avg_reward = total_reward / self.num_evaluation_episode
        return avg_reward
