# Actor-Critic Algorithm (AC)

## Related Paper

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## Game that this algorithm used

AC use  an open source reinforcement learning environment library called  [Gym](https://github.com/openai/gym) which is developed by OpenAI.

The game solved in AC from Gym is [**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/). "A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over."[1](https://gym.openai.com/envs/CartPole-v0/)

## How to run AC

Before running AC, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

After installation, you can directly use the following command to run the AC algorithm.

For comprehensive performance considerations on CPU, it is recommended to set `OMP_NUM_THREADS` and configure a unified configuration to 1/4 of the number of physical cores, such as export `OMP_NUM_THREADS=32`.

```python
> export OMP_NUM_THREADS=32
> python train.py
```

You will obtain outputs which is similar with the things below

```python
------------------------------------
Evaluation result in episode 0 is 9.600
------------------------------------
Episode 0, steps: 20.0, reward: 20.0
Episode 1, steps: 45.0, reward: 25.0
Episode 2, steps: 56.0, reward: 11.0
Episode 3, steps: 73.0, reward: 17.0
Episode 4, steps: 101.0, reward: 28.0
Episode 5, steps: 144.0, reward: 43.0
Episode 6, steps: 176.0, reward: 32.0
Episode 7, steps: 188.0, reward: 12.0
Episode 8, steps: 227.0, reward: 39.0
Episode 9, steps: 244.0, reward: 17.0
```

## Supported Platform

AC algorithm supports  GPU and CPU platfor
