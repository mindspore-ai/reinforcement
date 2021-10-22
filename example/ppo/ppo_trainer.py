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
"""PPO Trainer"""
import numpy as np
import matplotlib.pyplot as plt

import mindspore
import mindspore.nn as nn
from mindspore.common.api import ms_function
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore_rl.agent.trainer import Trainer

class PPOTrainer(Trainer):
    """This is the trainer class of PPO algorithm. It arranges the PPO algorithm"""
    def __init__(self, msrl, params=None):
        nn.Cell.__init__(self, auto_prefix=False)
        self.all_ep_r = []
        self.all_eval_ep_r = []
        self.zero = Tensor(0, mindspore.float32)
        self.assign = P.Assign()
        self.squeeze = P.Squeeze()
        self.mod = P.Mod()
        self.equal = P.Equal()
        self.less = P.Less()
        self.reduce_mean = P.ReduceMean()
        self.transpose = P.Transpose()
        self.duration = params['duration']
        self.batch_size = params['batch_size']
        self.eval_interval = params['eval_interval']
        self.num_eval_episode = params['num_eval_episode']
        self.metrics = params['metrics']
        super(PPOTrainer, self).__init__(msrl)

    def train(self, episode):
        """The main function which arranges the algorithm"""
        for i in range(episode):
            if i % self.eval_interval == 0:
                eval_reward = self.evaluation()
                eval_reward = eval_reward.asnumpy()
                if self.metrics:
                    self.all_eval_ep_r.append(eval_reward)
                print("-----------------------------------------")
                print(
                    f"Evaluation result in episode {i} is {eval_reward:.3f}")
                print("-----------------------------------------")
            _, training_reward = self.train_one_episode()
            if self.metrics:
                self.all_ep_r.append(training_reward.asnumpy())
            print(f"Episode {i}, steps: {self.duration}, "
                  f"reward: {training_reward.asnumpy():.3f}")

        if self.metrics:
            plt.plot(np.arange(len(self.all_ep_r)), self.all_ep_r)
            plt.plot(np.arange(0, len(self.all_ep_r), self.eval_interval), self.all_eval_ep_r)
            plt.legend(['training reward', 'evaluation reward'], loc='upper left')
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.show()

    @ms_function
    def train_one_episode(self):
        """the algorithm in one episode"""
        training_loss = self.zero
        training_reward = self.zero
        j = self.zero
        state, _ = self.msrl.agent_reset_collect()

        while self.less(j, self.duration):
            reward, new_state, action, miu, sigma = self.msrl.agent_act(state)
            self.msrl.replay_buffer_insert([state, action, reward, new_state, miu, sigma])
            state = new_state
            reward = self.reduce_mean(reward)
            training_reward += reward
            j += 1

        replay_buffer_elements = self.msrl.get_replay_buffer_elements(transpose=True, shape=(1, 0, 2))
        state_list = replay_buffer_elements[0]
        action_list = replay_buffer_elements[1]
        reward_list = replay_buffer_elements[2]
        next_state_list = replay_buffer_elements[3]
        miu_list = replay_buffer_elements[4]
        sigma_list = replay_buffer_elements[5]

        training_loss += self.msrl.agent_learn(
            (state_list, action_list, reward_list, next_state_list, miu_list,
             sigma_list))
        self.msrl.replay_buffer_reset()
        return training_loss, training_reward

    @ms_function
    def evaluation(self):
        """evaluation function"""
        total_eval_reward = self.zero
        num_eval = self.zero
        while num_eval < self.num_eval_episode:
            eval_reward = self.zero
            state, _ = self.msrl.agent_reset_eval()
            j = self.zero
            while self.less(j, self.duration):
                reward, state = self.msrl.agent_evaluate(state)
                reward = self.reduce_mean(reward)
                eval_reward += reward
                j += 1
            num_eval += 1
            total_eval_reward += eval_reward
        avg_eval_reward = total_eval_reward / self.num_eval_episode
        return avg_eval_reward
