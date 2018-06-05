# 工具模块

## [action.py](action.py)

执行游戏动作

## [Key.py](Key.py)

使用 Python 执行 DirectX 敲键

> GTAV 必须使用 DirectX 形式的键，不能使用 windows 默认的键

## [game.py](game.py)

- 通过本地端口获取游戏数据
- 回到起始点

## [model.py](model.py)

游戏数据在 Python 中的形式

- Vector 三维向量
- Entity 游戏中的物体
- Player 游戏中的玩家（人物）
- Vehicle 游戏中的交通工具（车辆）
- Data 游戏数据的顶层结构

> 详细属性请查看 model.py 中的注释

## [reward.py](reward.py)

根据游戏数据(Data 对象)来计算一次动作的 Reward

- calReward 计算 Reward

另有一些辅助计算的函数，文件文件内注释

## [screen.py](screen.py)

- screen_grab 截取屏幕上指定位置的图像
