# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore_rl.environment.multi_environment_wrapper import MultiEnvironmentWrapper
from mindspore.communication import init, get_rank


class _Workers():
    r'''
    The _Workers class is class for the distributed algorithms.

    Args:
        msrl (MSRL): The MSRL instance.
        fragment_list (dict): All the fragmets for distribution.
        episode (int): The eposide for each training.
    '''
    def __init__(self, msrl, fragment_list, duration, episode):
        self.rank_id = get_rank()
        self.fid = str(self.rank_id)
        print("Assign fragment ", self.fid, " on worker ", self.rank_id)
        self.worker = fragment_list[self.rank_id](msrl, self.rank_id, duration, episode)

    def run(self):
        print("Start fragment ", self.fid, " on worker ", self.rank_id)
        self.worker.run()
        print("Finish fragment ", self.fid)

class Session():
    """
    The Session is a class for running MindSpore RL algorithms.

    Args:
        alg_config (dict): the algorithm configuration or the deployment configuration of the algorithm.
        deploy_config (dict): the deployment configuration for distribution. Default: None.
            For more details of configuration of algorithm, please have a look at
            https://www.mindspore.cn/reinforcement/docs/zh-CN/master/custom_config_info.html
    """

    def __init__(self, alg_config, deploy_config=None):
        self.msrl = MSRL(alg_config, deploy_config)
        self.dist = False
        if deploy_config:
            if deploy_config['distributed']:
                self.dist = True
                self.worker_num = deploy_config['worker_num']
                self.config = deploy_config['config']

    def run(self, class_type=None, is_train=True, episode=0, duration=0, params=None, callbacks=None):
        """
        Execute the reinforcement learning algorithm.

        Args:
            class_type (Trainer): The class type of the algorithm's trainer class. Default: None.
            is_train (bool): Run the algorithm in train mode or eval mode. Default: True
            episode (int): The number of episode of the training. Default: 0.
            duration (int): The number of duration of the training. Default: 0.
            params (dict): The algorithm specific training parameters. Default: None.
            callbacks (list[Callback]): The callback list. Default: None.
        """

        if self.dist:
            init("nccl")
            from fragments import get_all_fragments
            fragment_list = get_all_fragments(self.msrl.num_actors)
            workers = _Workers(self.msrl, fragment_list, duration, episode)
            workers.run()
        else:
            if params is None:
                trainer = class_type(self.msrl)
            else:
                trainer = class_type(self.msrl, params)
            ckpt_path = None
            if params and 'ckpt_path' in params:
                ckpt_path = params['ckpt_path']
            if is_train:
                trainer.train(episode, callbacks, ckpt_path)
                print('training end')
            else:
                if ckpt_path:
                    trainer.load_and_eval(ckpt_path)
                    print('eval end')
                else:
                    print('Please provide a ckpt_path for eval.')

        if isinstance(self.msrl.collect_environment, MultiEnvironmentWrapper):
            if self.msrl.collect_environment.num_proc != 1:
                for collect_env in self.msrl.collect_environment.mpe_env_procs:
                    collect_env.terminate()

        if isinstance(self.msrl.eval_environment, MultiEnvironmentWrapper):
            if self.msrl.eval_environment.num_proc != 1:
                for eval_env in self.msrl.eval_environment.mpe_env_procs:
                    eval_env.terminate()
