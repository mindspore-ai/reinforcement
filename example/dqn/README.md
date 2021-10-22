# Deep Q-Learning (DQN)

## Related Paper

1. Mnih, Volodymyr, et al. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

2. Mnih, Volodymyr, *et al.* [Human-level control through deep reinforcement learning. *Nature* **518,** 529–533 (2015).](https://www.nature.com/articles/nature14236)

## Game that this algorithm used

DQN uses  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in DQN is [**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://gym.openai.com/envs/CartPole-v0/)

## How to run DQN

Before running DQN, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the DQN algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`.

```python
> export OMP_NUM_THREADS=32
> python train.py
```

You will obtain outputs which is similar with the things below

```python
-----------------------------------------
Evaluation result in episode 0 is 9.300
-----------------------------------------
Episode 0, steps: 10.0, reward: 10.0
Episode 1, steps: 48.0, reward: 38.0
Episode 2, steps: 71.0, reward: 23.0
Episode 3, steps: 80.0, reward: 9.0
Episode 4, steps: 90.0, reward: 10.0
Episode 5, steps: 102.0, reward: 12.0
Episode 6, steps: 112.0, reward: 10.0
Episode 7, steps: 122.0, reward: 10.0
Episode 8, steps: 131.0, reward: 9.0
Episode 9, steps: 140.0, reward: 9.0
-----------------------------------------
Evaluation result in episode 10 is 9.300
-----------------------------------------
Episode 10, steps: 149.0, reward: 9.0
Episode 11, steps: 158.0, reward: 9.0
Episode 12, steps: 168.0, reward: 10.0
Episode 13, steps: 176.0, reward: 8.0
Episode 14, steps: 188.0, reward: 12.0
Episode 15, steps: 196.0, reward: 8.0
Episode 16, steps: 205.0, reward: 9.0
Episode 17, steps: 214.0, reward: 9.0
Episode 18, steps: 225.0, reward: 11.0
Episode 19, steps: 235.0, reward: 10.0
```

## Supported Platform

DQN algorithm supports  GPU, CPU and Ascend platform
