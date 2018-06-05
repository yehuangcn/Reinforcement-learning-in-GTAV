# 网络

## ResNet

1.  使用 pytorch 自带的在 ImageNet 上训练的 resnet50

```py
    self.net = resnet50(pretrained=True)
```

2.  前 3 层 ResBlock 使用预训练权重，最后一层 ResBlock 权重可训练

```py
    for param in self.net.parameters():  
        # 前几层卷积权重使用Imagenet的训练结果
        param.requires_grad = False
    for param in self.net.layer4.parameters():     # 最后一卷积权重可训练
        param.requires_grad = True
```

3.  图像卷积的特征和游戏数据 链接起来

```py
    info = info.view(info.size()[0], -1)
    out = torch.cat([out, info], dim=1)  # 图像卷积的特征和游戏数据cat起来
```

4.  输出长度为 128 的特征向量

```py
    out = F.relu(self.fc2(out)) # 返回的特征shape(-1,128)
```

## RNNResnet

1.  使用 LSTM

```py
    self.lstm = nn.LSTM(128, 64, batch_first=True)
```

1.  使用 长度为 为 len_rnn_seq 的序列，在 ResNet 网络的输出值作为 LSTM 的输入

```py
    features = self.resnet.forward(inputs, infoes)
    out, _ = self.lstm(features)
```
