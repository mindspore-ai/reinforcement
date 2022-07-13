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
"""MonteCarloTreeSearch Class"""

#pylint: disable=W0235
import os
import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.ops import operations as P
import mindspore.nn.probability.distribution as msd
from mindspore.common import ms_function


# Node Enum Value
VANILLA = 0
# Tree Enum Value
COMMON = 0


class MCTS(nn.Cell):
    """
    Monte Carlo Tree Search(MCTS) is a general search algorithm for some kinds of decision processes,
    most notably those employed in software that plays board games, such as Go, chess. It was originally
    proposed in 2006. A general MCTS has four phases:
    1. Selection - selects the next node according to the selection policy (like UCT, RAVE, AMAF and etc.).
    2. Expansion - unless the selection reached a terminal state, expansion adds a new child node to
    the last node (leaf node) that is selected in Selection phase.
    3. Simulation - performs a complete random simulation of the game/problem to obtain the payoff.
    4. Backpropagation - propagates the payoff for all visited node.
    As the time goes by, these four phases of MCTS is evolved. AlphaGo introduced neural network to
    MCTS, which makes the MCTS more powerful.
    This class is a mindspore ops implementation MCTS. User can use provided MCTS algorithm, or develop
    their own MCTS by derived base class (MonteCarloTreeNode) in c++.

    Args:
        env (Environment): It must be the subclass of Environment.
        tree_type (enum): A enum value which states for tree type. Such as COMMON.
        node_type (enum): A enum value whhich states for node type. Such as VANILLA.
        max_action (int): The max action this game can perform. When the visited node in selection is less than
             max action, the remained place will be filled as -1. For example, in tic-tac-toe is 9. User can also
             set max_action = -1, then the selection ops will only return the last action performed. It depends on
             how user implements the environment.
        max_iteration (int): The max training iteration of MCTS.
        state_shape (tuple): A tuple which states for the shape of state.
        customized_func (nn.Cell): Some algorithm specific codes. For more detail, please have a look at
            documentation of AlgorithmBasedFunc.

    Examples:
        >>> mcts = MCTS()
    """

    def __init__(self, env, tree_type, node_type, max_action, max_iteration, state_shape, customized_func):
        super().__init__()
        current_path = os.path.dirname(os.path.normpath(os.path.realpath(__file__)))
        self.mcts_creation = ops.Custom("{}/libmcts.so:MctsCreation".format(current_path), (1,), ms.int64, "aot")
        if max_action != -1:
            self.mcts_selection = ops.Custom("{}/libmcts.so:MctsSelection".format(current_path),
                                             ((1,), (max_action,), (1,)), (ms.int64, ms.int32, ms.int32), "aot")
        else:
            self.mcts_selection = ops.Custom("{}/libmcts.so:MctsSelection".format(current_path),
                                             ((1,), (1,), (1,)), (ms.int64, ms.int32, ms.int32), "aot")
        self.mcts_expansion = ops.Custom("{}/libmcts.so:MctsExpansion".format(current_path), (1,), (ms.bool_), "aot")
        self.mcts_backpropagation = ops.Custom(
            "{}/libmcts.so:MctsBackpropagation".format(current_path), (1,), (ms.bool_), "aot")
        self.best_action = ops.Custom("{}/libmcts.so:BestAction".format(current_path), (1,), (ms.int32), "aot")

        self.update_node_outcome = ops.Custom(
            "{}/libmcts.so:UpdateOutcome".format(current_path), (1,), (ms.bool_), "aot")
        self.update_node_terminal = ops.Custom(
            "{}/libmcts.so:UpdateTerminal".format(current_path), (1,), (ms.bool_), "aot")
        self.update_state = ops.Custom("{}/libmcts.so:UpdateState".format(current_path), (1,), (ms.bool_), "aot")
        self.update_root_state = ops.Custom(
            "{}/libmcts.so:UpdateRootState".format(current_path), (1,), (ms.bool_), "aot")
        self.get_state = ops.Custom("{}/libmcts.so:GetState".format(current_path), state_shape, (ms.float32), "aot")
        self.destroy_tree = ops.Custom("{}/libmcts.so:DestroyTree".format(current_path), (1,), (ms.bool_), "aot")
        self.depend = P.Depend()

        # Add side effect annotation
        self.mcts_expansion.add_prim_attr("side_effect_mem", True)
        self.mcts_backpropagation.add_prim_attr("side_effect_mem", True)
        self.update_node_outcome.add_prim_attr("side_effect_mem", True)
        self.update_node_terminal.add_prim_attr("side_effect_mem", True)
        self.update_state.add_prim_attr("side_effect_mem", True)
        self.update_root_state.add_prim_attr("side_effect_mem", True)
        self.destroy_tree.add_prim_attr("side_effect_mem", True)

        self.zero = Tensor(0, ms.int32)
        self.zero_float = Tensor(0, ms.float32)
        self.true = Tensor(True, ms.bool_)
        self.false = Tensor(False, ms.bool_)

        self.env = env
        self.tree_type = tree_type
        self.node_type = node_type
        self.max_iteration = Tensor(max_iteration, ms.int32)
        self.max_action = max_action
        self.customized_func = customized_func
        temp_size = 1
        for shape in state_shape:
            temp_size *= shape
        self.state_size = temp_size

    @ms_function
    def mcts_search(self, *args):
        """
        mcts_search is the main function of MCTS. Invoke this function will return the best action of current
        state.

        Args:
            *args (Tensor): any values which will be the input of MctsCreation. Please following the table below
                to provide the input value.

                +------------------------------+--------------------------------------------------------+
                |  MCTS Node Type              |  Configuration Parameter    |  Notices                 |
                +==============================+========================================================+
                |  Vanilla                     |  UCT const                  |  UCT const is used to    |
                |                              |                             |  calculate UCT value in  |
                |                              |                             |  Selection phase         |
                +------------------------------+--------------------------------------------------------+

        Returns:
            action (mindspore.int32): The action which is returned by monte carlo tree search.
        """

        expanded = self.false
        reward = self.zero_float
        root_player = self.env.current_player()
        max_utility = self.env.max_utility()
        tree_handle = self.mcts_creation(self.tree_type, self.node_type, root_player,
                                         max_utility, self.state_size, *args)
        # Create a replica of environment
        new_state = self.env.save()
        self.update_root_state(tree_handle, new_state)
        i = self.zero
        while i < self.max_iteration:
            # 1. Interact with the replica of environment, and update the latest state and its reward
            visited_node, last_action, visited_path_length = self.mcts_selection(tree_handle, self.max_action)
            last_state = self.get_state(tree_handle, visited_node, visited_path_length-2)
            if expanded:
                self.env.load(last_state)
                new_state, reward, _ = self.env.step(last_action)
            else:
                new_state, reward, _ = self.env.load(last_state)
            self.update_state(tree_handle, visited_node, visited_path_length-1, new_state)
            # 2. Calculate the legal action and their probability of the latest state
            legal_action = self.env.legal_action()
            current_player = self.env.current_player()
            prior = self.customized_func.calculate_prior(new_state, legal_action)

            if not self.env.is_terminal():
                expanded = self.true
                self.mcts_expansion(tree_handle, self.node_type, visited_node,
                                    legal_action, reward, prior, current_player)
            else:
                self.update_node_outcome(tree_handle, visited_node, visited_path_length-1, reward)
                self.update_node_terminal(tree_handle, visited_node, visited_path_length-1, self.true)
            # 3. Calculate the return of the latest state, it could obtain from neural network
            #    or play randomly
            returns = self.customized_func.simulation(new_state)
            self.mcts_backpropagation(tree_handle, visited_node, returns)
            i += 1
        action = self.best_action(tree_handle)
        tree_handle = self.depend(tree_handle, action)
        self.destroy_tree(tree_handle)
        return action


class AlgorithmFunc(nn.Cell):
    """
    This is the base class for user to customize algorithm in MCTS. User need to inherit this base class
    and implement all the functions with SAME input and output.
    """
    def __init__(self):
        super().__init__()

    def calculate_prior(self, new_state, legal_action):
        """
        The functionality of calculate_prior is to calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment

        Returns:
            prior (mindspore.float32): The probability (or prior) of all the input legal actions.
        """
        raise NotImplementedError("You must implement this function")

    def simulation(self, new_state):
        """
        The functionality of simulation is to implement the simulation phase in MCTS. It takes the
        state as input and return the rewards.

        Args:
            new_state (mindspore.float32): The state of environment.

        Returns:
            rewards (mindspore.float32): The results of simulation.
        """
        raise NotImplementedError("You must implement this function")


class VanillaFunc(AlgorithmFunc):
    """
    This is the customized algorithm for VanillaMCTS. The prior of each legal action is uniform
    distribution and it plays randomly to obtain the result of simulation.
    """
    def __init__(self, env):
        super().__init__()
        self.minus_one = Tensor(-1, ms.int32)
        self.zero = Tensor(0, ms.int32)
        self.ones_like = P.OnesLike()
        self.categorical = msd.Categorical()
        self.env = env

        self.false = Tensor(False, ms.bool_)

    def calculate_prior(self, new_state, legal_action):
        """
        The functionality of calculate_prior is to calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment

        Returns:
            prior (mindspore.float32): The probability (or prior) of all the input legal actions.
        """
        invalid_action_num = (legal_action == -1).sum()
        prior = self.ones_like(legal_action).astype(ms.float32) / (len(legal_action) - invalid_action_num)
        return prior

    def simulation(self, new_state):
        """
        The functionality of calculate_prior is to calculate prior of the input legal actions.

        Args:
            new_state (mindspore.float32): The state of environment.
            legal_action (mindspore.int32): The legal action of environment

        Returns:
            prior (mindspore.float32): The probability (or prior) of all the input legal actions.
        """
        _, reward, done = self.env.load(new_state)
        while not done:
            legal_action = self.env.legal_action()
            mask = (legal_action == -1)
            invalid_action_num = (legal_action == -1).sum()
            prob = self.ones_like(legal_action).astype(ms.float32) / (len(legal_action) - invalid_action_num)
            prob[mask] = 0
            action = self.categorical.sample((), prob)
            new_state, reward, done = self.env.step(legal_action[action])
        return reward
