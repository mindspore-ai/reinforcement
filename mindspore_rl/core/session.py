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
Implementation of the session class.
"""
from mindspore_rl.core import MSRL

class Session():
    """
    The Session is a class for running MindSpore RL algorithms.

    Args:
        config (dict): the algorithm configuration or the deployment configuration of the algorithm.
            For more details of configuration of algorithm, please have a look at
            https://www.mindspore.cn/reinforcement/docs/zh-CN/r1.5/index.html
    """

    def __init__(self, config):
        self.msrl = MSRL(config)

    def run(self, class_type=None, episode=0, params=None):
        """
        Execute the reinforcement learning algorithm.

        Args:
            class_type (class type): The class type of the algorithm's trainer class. Default: None.
            episode (int): The number of episode of the training. Default: 0.
            params (dict): The algorithm specific training parameters. Default: None.
        """

        if class_type:
            if params is None:
                trainer = class_type(self.msrl)
            else:
                trainer = class_type(self.msrl, params)
            trainer.train(episode)
            print('training end')
