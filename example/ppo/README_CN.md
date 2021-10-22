# Proximal Policy Optimization (PPO)

## 相关论文

1. Schulman, John, et al. ["Proximal policy optimization algorithm"](https://arxiv.org/pdf/1707.06347.pdf)

## 使用的游戏

PPO算法使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在PPO中，解决了[**HalfCheetah-v2**](https://gym.openai.com/envs/HalfCheetah-v2/)这个来自Gym库的游戏。与于其他游戏如CartPole不同的是，这个游戏还依赖[MuJoCo](https://github.com/openai/mujoco-py)这个库。

## 如何运行PPO

在运行PPO前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.5.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.1,>=2.0

安装成功之后，可以直接通过输入如下指令来运行PPO。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`。

```python
> export OMP_NUM_THREADS=32
> python train.py
```

你会获得和下面内容相似的输出

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

## 支持平台

PPO算法支持GPU和CPU。
