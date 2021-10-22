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
'''Actor-Critic'''

from mindspore_rl.agent.learner import Learner
from mindspore_rl.agent.actor import Actor
import mindspore
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn.probability.distribution as msd
from mindspore.ops import operations as P

class ACPolicyAndNetwork():
    '''ACPolicyAndNetwork'''
    class ActorNet(nn.Cell):
        '''ActorNet'''
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.common = nn.Dense(input_size, hidden_size, bias_init=0.1)
            self.actor = nn.Dense(hidden_size, output_size, bias_init=0.1)
            self.relu = nn.LeakyReLU()
            self.softmax = P.Softmax()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            x = self.actor(x)
            return self.softmax(x)

    class CriticNet(nn.Cell):
        '''CriticNet'''
        def __init__(self, input_size, hidden_size, output_size=1):
            super().__init__()
            self.common = nn.Dense(input_size, hidden_size, bias_init=0.1)
            self.critic = nn.Dense(hidden_size, output_size, bias_init=0.1)
            self.relu = nn.LeakyReLU()

        def construct(self, x):
            x = self.common(x)
            x = self.relu(x)
            return self.critic(x)

    class ActorNNLoss(nn.Cell):
        '''Actor loss'''
        def __init__(self, actor_net):
            super().__init__(auto_prefix=False)
            self.actor_net = actor_net
            self.reduce_mean = ops.ReduceMean()
            self.log = ops.Log()
            self.neg = ops.Neg()
            self.softmax = ops.Softmax()
            self.expand_dims = ops.ExpandDims()
            self.cast = ops.Cast()

        def construct(self, state, td_error, a):
            action_probs_t = self.actor_net(self.expand_dims(state, 0))
            a_prob = action_probs_t[0][self.cast(a, mindspore.int32)]
            action_log_probs = self.log(a_prob)
            actor_loss = self.neg(self.reduce_mean(action_log_probs*td_error))
            return actor_loss

    class CriticNNLoss(nn.Cell):
        '''Critic loss'''
        def __init__(self, critic_net, gamma):
            super().__init__(auto_prefix=False)
            self._critic_net = critic_net
            self.square = ops.Square()
            self.gamma = gamma
            self.squeeze = ops.Squeeze()
            self.mul = ops.Mul()
            self.add = ops.Add()
            self.sub = ops.Sub()
            self.expand_dims = ops.ExpandDims()

        def construct(self, state, r, v_):
            v = self._critic_net(self.expand_dims(state, 0))
            v = self.squeeze(v)
            v_ = self.squeeze(v_)
            td_error = self.sub(self.add(r, self.mul(self.gamma, v_)), v)
            critic_loss = self.square(td_error)
            return critic_loss

    def __init__(self, params):
        self.actor_net = self.ActorNet(params['state_space_dim'], params['hidden_size'],
                                       params['action_space_dim'])
        self.critic_net = self.CriticNet(params['state_space_dim'], params['hidden_size'])
        optimizer_a = nn.Adam(self.actor_net.trainable_params(),
                              learning_rate=params['alr'])
        optimizer_c = nn.Adam(self.critic_net.trainable_params(),
                              learning_rate=params['clr'])
        actor_loss_net = self.ActorNNLoss(self.actor_net)
        self.actor_net_train = nn.TrainOneStepCell(actor_loss_net, optimizer_a)
        self.actor_net_train.set_train(mode=True)
        critic_loss_net = self.CriticNNLoss(self.critic_net, gamma=params['gamma'])
        self.critic_net_train = nn.TrainOneStepCell(critic_loss_net, optimizer_c)
        self.critic_net_train.set_train(mode=True)


class ACActor(Actor):
    '''AC Actor'''
    def __init__(self, params=None):
        super(ACActor, self).__init__()
        self._params_config = params
        self._environment = params['environment']
        self._eval_env = params['eval_environment']
        self._actor_net = params['actor_net']
        self.c_dist = msd.Categorical(dtype=mindspore.float32)
        self.expand_dims = P.ExpandDims()
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        self.argmax = P.Argmax(output_type=mindspore.int32)

    def act(self, state):
        '''Sample action to act in env'''
        ts0 = self.expand_dims(state, 0)
        action_probs_t = self._actor_net(ts0)
        action = self.reshape(self.c_dist.sample((1,), probs=action_probs_t), (1,))
        new_state, reward, done = self._environment.step(self.cast(action, mindspore.int32))
        return done, reward, new_state, action

    def evaluate(self, state):
        """Evaluate the trained policy"""
        ts0 = self.expand_dims(state, 0)
        action_probs_t = self._actor_net(ts0)
        action = self.reshape(self.argmax(action_probs_t), (1,))
        new_state, reward, done = self._eval_env.step(self.cast(action, mindspore.int32))
        return done, reward, new_state

class ACLearner(Learner):
    '''AC Learner'''
    def __init__(self, params):
        super(ACLearner, self).__init__()
        self._params_config = params
        self.gamma = Tensor(self._params_config['gamma'], mindspore.float32)
        self._actor_net_train = params['actor_net_train']
        self._critic_net_train = params['critic_net_train']
        self._actor_net = params['actor_net']
        self._critic_net = params['critic_net']
        self.sub = ops.Sub()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.reshape = ops.Reshape()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = P.Squeeze()

    def learn(self, samples):
        # Calculate the td_error
        state = samples[0]
        r = samples[1]
        state_ = samples[2]
        a = samples[3]
        v_ = self._critic_net(self.expand_dims(state_, 0))
        v = self._critic_net(self.expand_dims(state, 0))
        v_ = self.squeeze(v_)
        v = self.squeeze(v)
        td_error = self.sub(self.add(r, self.mul(self.gamma, v_)), v)
        critic_loss = self._critic_net_train(state, r, v_)
        actor_loss = self._actor_net_train(state, td_error, a)
        return actor_loss + critic_loss
