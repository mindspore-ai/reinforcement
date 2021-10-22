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
"""AC Trainer"""
import matplotlib.pyplot as plt

from mindspore_rl.agent.trainer import Trainer
import mindspore
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter

class ACTrainer(Trainer):
    '''ACTrainer'''
    def __init__(self, msrl, params):
        nn.Cell.__init__(self, auto_prefix=False)
        self.evaluation_interval = params['evaluation_interval']
        self.num_evaluation_episode = params['num_evaluation_episode']
        self.metrics = params['metrics']
        self._total_reward = Parameter(Tensor(0, mindspore.float32), name='reward', requires_grad=False)
        self.zero = Parameter(Tensor(0, mindspore.float32))
        self.one = Parameter(Tensor(1, mindspore.float32))
        self.done_r = Parameter(Tensor([-20.0], mindspore.float32))
        self.zero_value = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.less = P.Less()
        self.rewards_for_save = []
        self.all_ep_r = []
        self.all_steps = []
        super(ACTrainer, self).__init__(msrl)

    def train(self, episode):
        steps = 0
        for i in range(episode):
            if i % self.evaluation_interval == 0:
                reward = self.evaluation()
                print("-----------------------------------------")
                print(f'Evaluation result in episode {i} is {(reward.asnumpy()):.3f}')
                print("-----------------------------------------")
                if self.metrics:
                    self.all_steps.append(steps)
                    self.all_ep_r.append(reward.asnumpy())
            total_reward, episode_steps = self.train_one_episode()
            steps += episode_steps.asnumpy()
            r = total_reward.asnumpy().tolist()
            if self.metrics:
                if i == 0:
                    running_reward = r
                else:
                    running_reward = r * 0.05 + running_reward * 0.95
                self.rewards_for_save.append(running_reward)
            print(f'Episode {i}, steps: {steps}, reward: {r:.3f}')
        reward = self.evaluation()
        print("-----------------------------------------")
        print(f'Evaluation result in episode {episode} is {(reward.asnumpy()):.3f}')
        print("-----------------------------------------")
        if self.metrics:
            self.all_ep_r.append(reward.asnumpy())
            self.all_steps.append(steps)
            self.plot()


    @ms_function
    def train_one_episode(self):
        '''Train one episode'''
        state, _ = self.msrl.agent_reset_collect()
        self.assign(self._total_reward, self.zero)
        steps = self.zero
        while True:
            done, r, state_, a = self.msrl.agent_act(state)
            r = self.squeeze(r)
            self._total_reward += r
            if done:
                r = self.done_r
            self.msrl.agent_learn([state, r, state_, a])
            steps += 1
            state = state_
            if done:
                break
        return self._total_reward, steps

    @ms_function
    def evaluation(self):
        '''evaluation'''
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

    def plot(self):
        plt.plot(self.all_steps, self.all_ep_r)
        plt.xlabel('eposide')
        plt.ylabel('reward')
        plt.savefig('ac_rewards.png')
