from torchvision.models import resnet50
import torch.nn.functional as F
from torch.nn import Module
import torch
from torch import nn
from torch.autograd import Variable
from constant.model import (near_by_peds_limit,
                            near_by_props_limit,
                            near_by_touching_peds_limit,
                            near_by_touching_props_limit,
                            near_by_touching_vehicles_limit,
                            near_by_vehicles_limit)
import numpy as np
dtype = torch.float32
device = torch.device("cuda:0")

len_rnn_seq = 3


class ResNet(Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = resnet50(pretrained=True)
        for param in self.net.parameters():  # 前几层卷积权重使用Imagenet的训练结果
            param.requires_grad = False
        for param in self.net.layer4.parameters():  # 最后一卷积权重可训练
            param.requires_grad = True
        self.net.fc = nn.Linear(2048 * 2 * 2, 256)  # 最后一层卷积shape (-1,2048,2,2)
        info_len = 9 + 5 * 3 + (  # 游戏数据的长度
            near_by_peds_limit +
            near_by_props_limit +
            near_by_touching_peds_limit +
            near_by_touching_props_limit +
            near_by_touching_vehicles_limit +
            near_by_vehicles_limit
        ) * 3
        self.fc2 = nn.Linear(256 + info_len, 128)

    def forward(self, input_tensor, info):
        out = self.net.forward(input_tensor)
        info = info.view(info.size()[0], -1)
        out = torch.cat([out, info], dim=1)  # 图像卷积的特征和游戏数据cat起来
        out = F.relu(self.fc2(out)) # 返回的特征shape(-1,128)

        return out


class RNNResnet(Module):
    def __init__(self):
        super(RNNResnet, self).__init__()
        self.resnet = ResNet()
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.action_fc3 = nn.Linear(64, 6)
        self.state_fc3 = nn.Linear(64, 1)

    def forward(self, inputs, infoes):
        features = self.resnet.forward(inputs, infoes)
        if len(features) >= len_rnn_seq:
            assert len(features) % len_rnn_seq == 0
            features = features.view(
                (len(features) // sd, len_rnn_seq, -1))
            out, _ = self.lstm(features)
        else:
            features = features.view(1, features.size()[0], features.size()[1])
            out, _ = self.lstm(features)

        out = torch.chunk(out, len_rnn_seq-1, dim=1)[-1]
        out = torch.squeeze(out, 1)
        policy = F.softmax(self.action_fc3(out), dim=1)
        state = self.state_fc3(out)
        return policy, state
