# Proximal Policy Optimization (PPO)

## Related Paper

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## Game that this algorithm used

PPO uses  an open source reinforcement learning environment library called [Gym](https://github.com/openai/gym), which is developed by OpenAI.

The game solved in PPO is called [**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/), it is from Gym, but this game depends on an advanced physics simulation called [MuJoCo](https://github.com/openai/mujoco-py).

## How to run PPO

Before running PPO, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.1,>=2.0

After installation, you can directly use the following command to run the PPO algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`.

```python
> export OMP_NUM_THREADS=32
> python train.py
```

You will obtain outputs which is similar with the things below

```python
-----------------------------------------
Evaluation result in episode 0 is -0.284
-----------------------------------------
Episode 0, steps: 1000, reward: -165.2470
Episode 1, steps: 1000, reward: -118.340
Episode 2, steps: 1000, reward: -49.168
Episode 3, steps: 1000, reward: 11.567
Episode 4, steps: 1000, reward: 70.142
Episode 5, steps: 1000, reward: 131.269
Episode 6, steps: 1000, reward: 198.397
Episode 7, steps: 1000, reward: 232.347
Episode 8, steps: 1000, reward: 260.215
Episode 9, steps: 1000, reward: 296.540
Episode 10, steps: 1000, reward: 314.946
Episode 11, steps: 1000, reward: 315.864
Episode 12, steps: 1000, reward: 442.881
Episode 13, steps: 1000, reward: 477.359
Episode 14, steps: 1000, reward: 461.781
Episode 15, steps: 1000, reward: 448.245
Episode 16, steps: 1000, reward: 534.768
Episode 17, steps: 1000, reward: 535.590
Episode 18, steps: 1000, reward: 568.602
Episode 19, steps: 1000, reward: 637.380
-----------------------------------------
Evaluation result in episode 20 is 782.709
-----------------------------------------
```

## Supported Platform

PPO algorithm supports  GPU and CPU platform.
