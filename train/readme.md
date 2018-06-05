# 训练

1.  在 GPU 上运行

```py
    device = torch.device("cuda:0")
```

2.  默认的 Type 是 float32

```py
    dtype = torch.float32
```

3.  图像转换方法

`channel last` 转换成 `channel first`，像素值转换为 `0-1` 之间

```py
    my_transformer = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
```

## getScreenPreprocess

截取屏幕图像并预处理,使用`my_transformer`转换

## getReward

计算 Reward

- data 当前帧的游戏数据
- previous_dataes 前几帧的游戏数据

> - 如果没有前几帧的游戏数据，那就只计算这一帧的 reward

> - 如果又前几帧的游戏数据，就计算下上一帧和这一帧车辆是否有损坏

## getAssistInfo

对游戏游戏数据(data)进行处理，筛选并计算出要进入 Resnet 的数据

## retriveInputAgainstException

    从本地端口获取游戏数据(data)，并处理为要进入ResNet 的 数据(info)
    返回 data, info

## train

    训练函数

### 核心逻辑

1.  每次在游戏中执行 act_batch_num 步

    1.  每一步查看当前状态，
        - 空转(撞上车辆/墙壁等) ---- 回合结束 负分
        - 偏离航线 ---- 回合结束 负分
        - 远离终点 ---- 回合结束 负分
        - 到达终点 ---- 回合结束 高分
    2.  如果回合结束就回到起点
    3.  每一步保存图像 `x`,`data`,`info`,`reward` 到 `buffer` 中

1.  游戏中执行 `act_batch_num` 步之后，从 `buffer` 中抽取 前 `learn_batch_num` 个记录进行训练
    计算 `state` 时跨 `n` 步

    - state_loss ----- 计算值函数的损失
    - policise_loss ----- 计算策略函数的损失
    - loss = state_loss + policise_loss ----- 损失函数相加
    - loss.backward() ----- 计算梯度
    - torch.nn.utils.clip_grad_norm\_(train_parameters, - 0.5) ----- 梯度清理

1.  按训练次数 x 保存网络的 state_dict

## class Record

    放入`buffer`的记录类
