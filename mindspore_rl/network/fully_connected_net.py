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
FullyConnectedNet.
"""

import mindspore.nn as nn


class FullyConnectedNet(nn.Cell):
    """
    A basic fully connected neural network.

    Args:
        input_size(int): numbers of input size.
        hidden_size(int): numbers of hidden layers.
        output_size(int): numbers of output size.

    Examples:
        >>> input = Tensor(np.ones([2, 4]).astype(np.float32))
        >>> net = FullyConnectedNet(4, 10, 2)
        >>> output = net(input)
        >>> print(output.shape)
        (2, 2)
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNet, self).__init__()
        self.linear1 = nn.Dense(
            input_size,
            hidden_size,
            weight_init="XavierUniform")
        self.linear2 = nn.Dense(
            hidden_size,
            output_size,
            weight_init="XavierUniform")
        self.relu = nn.ReLU()

    def construct(self, x):
        """
        Returns output of Dense layer.

        Args:
            x (Tensor): Tensor as the input of network.

        Returns:
            The output of the Dense layer.
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x
