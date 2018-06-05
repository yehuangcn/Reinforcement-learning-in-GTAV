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
