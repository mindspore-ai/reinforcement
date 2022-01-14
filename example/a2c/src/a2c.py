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
'''Advantage Actor Critic'''

from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.actor import Actor
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
from mindspore.ops import operations as P
import numpy as np

seed = 42
np.random.seed(seed)


class A2CPolicyAndNetwork():
    '''A2CPolicyAndNetwork'''
    class ActorCriticNet(nn.Cell):
        '''ActorCriticNet'''

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.common = nn.Dense(
                input_size, hidden_size, weight_init='XavierUniform')
            self.actor = nn.Dense(hidden_size, output_size,
                                  weight_init='XavierUniform')
            self.critic = nn.Dense(hidden_size, 1, weight_init='XavierUniform')
            self.relu = nn.LeakyReLU()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            return self.actor(x), self.critic(x)

    class Loss(nn.Cell):
        '''Actor-Critic loss'''

        def __init__(self, a2c_net):
            super().__init__(auto_prefix=False)
            self.a2c_net = a2c_net
            self.reduce_sum = ops.ReduceSum(keep_dims=False)
            self.log = ops.Log()
            self.gather = ops.GatherD()
            self.softmax = ops.Softmax()
            self.minimum = ops.Minimum()
            self.abs = ops.Abs()
            self.expand_dims = ops.ExpandDims()
            self.add = ops.Add()
            self.square = ops.Square()
            self.mul = ops.Mul()
            self.delta = Tensor(1, mindspore.float32)
            self.h_d = Tensor(0.5, mindspore.float32)

        def construct(self, states, actions, returns):
            '''Calculate actor loss and critic loss'''
            action_logits_ts, values = self.a2c_net(states)
            action_probs_t = self.softmax(action_logits_ts)
            action_probs = self.gather(action_probs_t, 1, actions)
            advantage = returns - values
            action_log_probs = self.log(action_probs)
            adv_mul_prob = action_log_probs * advantage
            actor_loss = -self.reduce_sum(adv_mul_prob)
            q = self.minimum(self.abs(advantage), self.delta)
            l = self.abs(advantage) - q
            loss = self.add(self.mul(self.h_d, self.mul(q, q)), l)
            critic_loss = self.reduce_sum(loss)
            return critic_loss + actor_loss

    def __init__(self, params):
        self.a2c_net = self.ActorCriticNet(params['state_space_dim'], params['hidden_size'],
                                           params['action_space_dim'])
        optimizer = nn.Adam(self.a2c_net.trainable_params(),
                            learning_rate=params['lr'])
        loss_net = self.Loss(self.a2c_net)
        self.a2c_net_train = nn.TrainOneStepCell(loss_net, optimizer)
        self.a2c_net_train.set_train(mode=True)

#pylint: disable=W0223


class A2CActor(Actor):
    '''A2C Actor'''

    def __init__(self, params=None):
        super(A2CActor, self).__init__()
        self._params_config = params
        self.a2c_net = params['a2c_net']
        self._environment = params['environment']
        self.c_dist = msd.Categorical(dtype=mindspore.float32, seed=seed)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.softmax = ops.Softmax()
        self.zero = Tensor(0, mindspore.int64)
        self.loop_size = Tensor(1000, mindspore.int64)
        self.done = Tensor(True, mindspore.bool_)
        self.states = nn.TensorArray(mindspore.float32, (4,))
        self.actions = nn.TensorArray(mindspore.int32, (1,))
        self.rewards = nn.TensorArray(mindspore.float32, (1,))
        self.print = P.Print()

    def act(self, phase, params):
        '''Store returns into TensorArrays from env'''
        if phase == 2:
            t = self.zero
            while t < self.loop_size:
                self.states.write(t, params)
                ts0 = self.expand_dims(params, 0)
                action_logits, _ = self.a2c_net(ts0)
                action_probs_t = self.softmax(action_logits)
                action = self.reshape(self.c_dist.sample(
                    (1,), probs=action_probs_t), (1,))
                action = self.cast(action, mindspore.int32)
                self.actions.write(t, action)
                new_state, reward, done = self._environment.step(action)
                self.rewards.write(t, reward)
                params = new_state
                if done == self.done:
                    break
                t += 1
            rewards = self.rewards.stack()
            states = self.states.stack()
            actions = self.actions.stack()
            self.rewards.clear()
            self.states.clear()
            self.actions.clear()
            return rewards, states, actions
        self.print("Phase is incorrect")
        return 0


class A2CLearner(Learner):
    '''A2C Learner'''

    def __init__(self, params):
        super(A2CLearner, self).__init__()
        self._params_config = params
        self.gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        self._a2c_net_train = params['a2c_net_train']
        self.shape = ops.DynamicShape()
        self.moments = nn.Moments(keep_dims=False)
        self.sqrt = ops.Sqrt()
        self.zero = Tensor(0, mindspore.int64)
        self.epsilon = Tensor(1.1920929e-07, mindspore.float32)
        self.zero_float = Tensor(0.0, mindspore.float32)
        self.returns = nn.TensorArray(mindspore.float32, (1,))

    def learn(self, experience):
        '''Calculate the loss and update'''
        rewards = experience[0]
        states = experience[1]
        actions = experience[2]
        n = self.shape(rewards)[0]
        iter_num = self.zero
        discounted_sum = self.zero_float
        while iter_num < n:
            i = n - iter_num - 1
            reward = rewards[i]
            discounted_sum = reward + self.gamma * discounted_sum
            self.returns.write(i, discounted_sum)
            iter_num += 1
        returns = self.returns.stack()
        self.returns.clear()
        adv_mean, adv_var = self.moments(returns)
        normalized_returns = (returns - adv_mean) / \
            (self.sqrt(adv_var) + self.epsilon)
        a2c_loss = self._a2c_net_train(states, actions, normalized_returns)
        return a2c_loss
