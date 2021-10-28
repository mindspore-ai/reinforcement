# Deep Q-Learning (DQN)

## 相关论文

1. Mnih, Volodymyr, et al. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

2. Mnih, Volodymyr, *et al.* [Human-level control through deep reinforcement learning. *Nature* **518,** 529–533 (2015).](https://www.nature.com/articles/nature14236)

## 使用的游戏

DQN算法使用了OpenAI开发的一个强化学习环境库[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在DQN算法中，解决了倒立摆([**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/))游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://gym.openai.com/envs/CartPole-v0/)

## 如何运行DQN

在运行DQN前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行DQN。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`。

```python
> export OMP_NUM_THREADS=32
> python train.py
```

你会获得和下面内容相似的输出

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

## 支持平台

DQN算法支持GPU和CPU。
