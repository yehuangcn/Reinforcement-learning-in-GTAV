# Reinforcement-learning-in-GTA V

在 GTAV 环境 中使用强化学习算法(Actor-Critic-LSTM)

## 配置

1.  Grand Theft Auto V(侠盗猎车手 5) steam 或者原装均可
2.  numpy Pytorch(gp 版本) 推荐最新版本(0.40) torchvision
3.  GPU(推荐 GTX 960 及以上)，我使用的 GTX 960 同时运行 GTA V 和 Actor-Critic 算法有点吃力
4.  系统 Windows，因为MAC 以及 linux 上没有 GTA V
5.  [GTAV 强化学习环境](https://github.com/zhaoying9105/GTAV-RewardHook)，我的另一个项目。

## 文件结构

- constant  [代码详解](constant/readme.md)
- 用于算法的一些常数，包括按键，日志，网络常量，游戏画面截取位置
- model  [代码详解](model/readme.md)
- 网络，使用预训练的 ResNet 得到卷积特征的序列，然后进入 LSTM 得到策略函数和值函数。
- train  [代码详解](train/readme.md)
- 训练过程
- util  [代码详解](util/readme.md)
- 工具函数，包括游戏操作，游戏数据获获取、转换和计算，游戏画面截取

## 训练结果

训练目标 使用 GTAV 强化学习环境 提供的起点和终点，希望 agent 学习到从起点不碰撞行驶到终点的策略：
训练结果 能够主动避开车辆和墙壁，但是对路线不敏感
