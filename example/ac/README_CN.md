# Actor-Critic Algorithm (AC)

## 相关论文

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## 使用的游戏

AC使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在AC中，解决了倒立摆（[**CartPole-v0**](https://gym.openai.com/envs/CartPole-v0/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://gym.openai.com/envs/CartPole-v0/)

## 如何运行AC

在运行AC前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行AC。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`。

```python
> export OMP_NUM_THREADS=32
> python train.py
```

你会获得和下面内容相似的输出

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

## 支持平台

AC算法支持GPU和CPU。
