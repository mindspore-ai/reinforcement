# Proximal Policy Optimization (PPO)

## Related Paper

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## Game that this algorithm used

PPO uses  an open source reinforcement learning environment library called [Gym](https://github.com/openai/gym), which is developed by OpenAI.

The game solved in PPO is called [**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/), it is from Gym, but this game depends on an advanced physics simulation called [MuJoCo](https://github.com/openai/mujoco-py).

## How to run PPO

Before running PPO, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

After installation, you can directly use the following command to run the PPO algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`(edit in `run_standalone_train.sh`).

### Train

```shell
> cd example/ppo/scripts
> bash run_standalone_train.sh [EPISODE](optional) [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ppo_train_log.txt`.

```shell
Episode 0, loss is 90.738, reward is -165.2470.
Episode 1, loss is 26.225, reward is -118.340.
Episode 2, loss is 28.407, reward is -49.168.
Episode 3, loss is 46.766, reward is 11.567.
Episode 4, loss is 81.542, reward is 70.142.
Episode 5, loss is 101.262, reward is 131.269.
Episode 6, loss is 95.097, reward is 198.397.
Episode 7, loss is 97.289, reward is 232.347.
Episode 8, loss is 133.97, reward is 260.215.
Episode 9, loss is 112.115, reward is 296.540.
Episode 10, loss is 123.75, reward is 314.946.
Episode 11, loss is 140.75, reward is 315.864.
Episode 12, loss is 157.439, reward is 442.881.
Episode 13, loss is 217.987, reward is 477.359.
Episode 14, loss is 199.457, reward is 461.781.
Episode 15, loss is 194.124, reward is 448.245.
Episode 16, loss is 199.476, reward is 534.768.
Episode 17, loss is 210.211, reward is 535.590.
Episode 18, loss is 207.483, reward is 568.602.
Episode 19, loss is 216,794, reward is 637.380.
-----------------------------------------
Evaluate result for episode 20 total rewards is 782.709
-----------------------------------------
```

### Eval

```shell
> cd example/ppo/scripts
> bash run_standalone_eval.sh [CKPT_FILE_PATH] [DEVICE_TARGET](optional)
```

You will obtain outputs which is similar with the things below in `ppo_eval_log.txt`.

```shell
Load file /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
Evaluate result is 6000.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## Supported Platform

PPO algorithm supports  GPU and CPU platform.
