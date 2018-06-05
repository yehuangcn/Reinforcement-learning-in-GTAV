import logging
import math
import random
import sys
sys.path.append('')
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms

from constant.log import logger
from constant.model import (near_by_peds_limit,
                            near_by_props_limit,
                            near_by_touching_peds_limit,
                            near_by_touching_props_limit,
                            near_by_touching_vehicles_limit,
                            near_by_vehicles_limit)
from constant.screen import screen_pos
from model.resnet import RNNResnet
from util import count_down
from util.game import backToStartPoint, gameData, getDisctanceFromPointToLine
from util.action import control_by_vec, random_action
from util.model import Data
from util.reward import (calReward, nearbyVectors, distance,
                         includedAngle, minus,
                         multiply, vector2Numpy, vectors2Numpy)
from util.screen import screen_grab


device = torch.device("cuda:0")
dtype = torch.float32

my_transformer = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def getScreenPreprocess():
    """截取屏幕图像并预处理
    Returns:
        torch.cuda.FloatTensor -- 图像
    """

    x = screen_grab(screen_pos)  # 在指定位置截取指定大小的图像
    x = np.array(x)
    x = cv2.resize(x, (256, 256))  # resize 为 (256,256,3)
    x = my_transformer(x)  # 转换为(3,256,256),然后使得元素的值在（0，1）之间
    x = x.unsqueeze(0)  # (3，256，256 )=> (1,3,256,256)
    x = x.cuda(device=device)  # cpu => gpu
    return x


def getReward(data, previous_dataes):
    """计算Reward
    如果没有前几帧的游戏数据，那就只计算这一帧的reward
    如果又前几帧的游戏数据，就计算下上一帧和这一帧车辆是否有损坏
    note: 还有很大的改进余地

    Arguments:
        data {Data} -- 当前帧的游戏数据
        previous_dataes {list[Data]} -- 前几帧的游戏数据

    Returns:
        float -- Reward
    """

    if len(previous_dataes) > 0:
        health_reward = 0
        if data.car.Health - previous_dataes[0].car.Health < 0:
            health_reward = (data.car.Health -
                             previous_dataes[0].car.Health) * 2
        reward = calReward(data) + health_reward
    else:
         # 如果没有前几帧的数据
        reward = calReward(data)
    return reward


def getAssistInfo(data):
    """
    对游戏游戏数据(data)进行处理，筛选并计算出要进入 Resnet 的数据
    
    Arguments:
        data {Data} -- 这一帧的游戏数据
    
    Returns:
        [list] -- 输入ResNet的数据
    """

    assist = []
    speed = data.car.Speed
    acceleration = data.car.Acceleration
    target_direction = minus(
        data.car.Position, data.next_position_on_street)
    distanceToEndPosition = distance(
        data.car.Position, data.endPosition)
    angle = includedAngle(target_direction, data.forward_vector3)
    endPositionVector = vector2Numpy(
        minus(data.car.Position, data.endPosition))
    near_by_vehicles = vectors2Numpy(nearbyVectors(data.car.Position, [
        item.Position for item in data.near_by_vehicles], near_by_vehicles_limit))
    near_by_peds = vectors2Numpy(nearbyVectors(data.car.Position, [
        item.Position for item in data.near_by_peds if item.IsOnFoot], near_by_peds_limit))
    near_by_props = vectors2Numpy(nearbyVectors(data.car.Position, [
        item.Position for item in data.near_by_props], near_by_props_limit))
    near_by_touching_vehicles = vectors2Numpy(nearbyVectors(
        data.car.Position, [item.Position for item in data.near_by_touching_vehicles], near_by_touching_vehicles_limit))
    near_by_touching_peds = vectors2Numpy(nearbyVectors(
        data.car.Position, [item.Position for item in data.near_by_touching_peds], near_by_touching_peds_limit))
    near_by_touching_props = vectors2Numpy(nearbyVectors(
        data.car.Position, [item.Position for item in data.near_by_touching_props], near_by_touching_props_limit))
    assist.append(speed)  # 当前速度
    assist.append(angle)  # 速度和终点方向的夹角
    assist.append(acceleration)  # 加速度 0 1 -1
    assist.append(data.car.SteeringScale)  # 转向
    assist.append(data.car.SteeringAngle)  # 转向角
    assist.append(data.car.CurrentRPM)  # 每分钟转速
    assist.append(1 if data.car.IsOnScreen else 0)  # 车辆是否出现在屏幕上
    assist.append(1 if data.onRoad else 0)  # 车辆是否在道路上(不在人行道或者道路中间的绿化上)
    assist.append(distanceToEndPosition)  # 到终点的直线距离
    assist.extend(vector2Numpy(data.car.Velocity))  # 速度向量
    assist.extend(vector2Numpy(data.car.ForwardVector))  # 车头的方向向量
    assist.extend(vector2Numpy(data.car.RightVector))  # 车右方的方向向量
    assist.extend(vector2Numpy(data.car.Rotation))  # 车辆旋转角度
    assist.extend(endPositionVector.flatten())  # 以当前位置为原点，终点的方向向量

    assist.extend(near_by_vehicles.flatten())  # 以当前位置为原点，附近的车辆的位置
    assist.extend(near_by_peds.flatten())  # 以当前位置为原点，附近的人的位置
    # 以当前位置为原点，附近的物体的位置(可移动和可撞毁的物体，不包括建筑，墙壁，电线杆等不可移动物体，包括路灯/铁丝网/消防栓等可撞毁的物体)
    assist.extend(near_by_props.flatten())
    # 以当前位置为原点，附近有接触(碰撞)的车辆的位置
    assist.extend(near_by_touching_vehicles.flatten())
    assist.extend(near_by_touching_peds.flatten())  # 以当前位置为原点，附近有接触(碰撞)的人的位置
    assist.extend(near_by_touching_props.flatten())  # 以当前位置为原点，附近有接触(碰撞)的物体的位置
    assist = np.array(assist)
    assist = np.reshape(assist, newshape=(1, len(assist)))
    return assist


def retriveInputAgainstException():
    """获取游戏数据
    json = gameData(fromNet=True) 有可能出错，所以需要try-except 持续获取
    Returns:
        data -- Data类型
        info -- 根据 data 处理得到的数据
    """

    while True:
        try:
            json = gameData(fromNet=True)  # 从网络获取Json形式(dict)的游戏数据
            data = Data(json)  # 游戏数据从 dict 转换为Data对象
            info = getAssistInfo(data)  # 根据 data 处理得到的数据
            info = torch.tensor(info, dtype=dtype, device=device)
            return data, info
        except Exception as e:
            continue


def train():
    counter = 1
    net = RNNResnet().cuda()
    train_parameters = [param for param in net.parameters()
                        if param.requires_grad == True]
    optimizer = optim.SGD(train_parameters, lr=0.001, momentum=0.9)
    # 倒计时
    count_down(1)  # 倒计时 两秒钟
    buffer = []  # 记录回放
    max_buff_num = 500  # 记录回访最大容量
    act_batch_num = 20  # 每次训练之前在线执行次数
    goodEpisode = 0  # 成功完成回合的次数
    badEposode = 0  # 回合失败的次数
    learn_batch_num = 24  # 每次训练时batch的大小
    num_step_to_check_trapped = 6  # 没 6 次检查车轮是否空转(trapped)
    speeds = []  # 车辆速度列表 长度为 num_step_to_check_trapped
    wheel_speeds = []  # 车轮速度列表 长度为 num_step_to_check_trapped
    gamma = 0.97  # reward discount
    n = 16  # 对这样长的序列计算  discount  的 reward
    least_buffer_for_learning = learn_batch_num + n  # buffer里面至少有这么多才进行训练
    len_rnn_seq = 3  # 进入 rnn 网络的序列长度
    previous_dataes = []  # 用于上一帧的预测的 data 列表 ，长度为  len_rnn_seq
    previous_xes = []  # 用于上一帧的预测的 截屏(x) 列表 ，长度为  len_rnn_seq
    previous_infoes = []  # 用于上一帧的预测的 info(辅助信息) 列表 ，长度为  len_rnn_seq
    dataes = []  # 用于这一帧的预测的 data 列表 ，长度为  len_rnn_seq
    xes = []  # 用于这一帧的预测的 截屏(x) 列表 ，长度为  len_rnn_seq
    infoes = []  # 用于这一帧的预测的 info(辅助信息) 列表 ，长度为  len_rnn_seq
    num_step_from_last_down = 0  # 距离上次项目结束已经有多少个step 了(不是counter)
    max_num_step_a_episode = 1500

    while True:
        print("counter " + str(counter))
        for i in range(act_batch_num):  # 每一个counter 运行  act_batch_num 次
            if len(dataes) > 0:
                if len(previous_dataes) >= len_rnn_seq:
                    previous_dataes.pop()  # 去掉最旧的数据(第一个数据)
                previous_dataes.append(data)  # 新数据放在最后
                if len(previous_xes) >= len_rnn_seq:
                    previous_xes.pop()
                previous_xes.append(x)
                if len(previous_infoes) >= len_rnn_seq:
                    previous_infoes.pop()
                previous_infoes.append(info)
                with torch.no_grad():  # 仅预测，不训练
                    actions, state = net.forward(
                        torch.cat(previous_xes, dim=0), torch.cat(previous_infoes, dim=0))
                rand_num = random.random()
                if rand_num > 0.9:  # 10 % 概率随机探索
                    actions = random_action()
                    control_by_vec(list(actions[0]), duration=0.2)  # 执行动作
                else:
                    control_by_vec(
                        list(actions[0].cpu().data.numpy()), duration=0.2)  # 执行动作
                time.sleep(0.1)  # 游戏执行动作的时间
                x = getScreenPreprocess()  # 动作执行之后的图像
                if len(xes) >= len_rnn_seq:
                    xes.pop()
                xes.append(x)
                data, info = retriveInputAgainstException()  # 动作执行之后的游戏数据
                if len(dataes) >= len_rnn_seq:
                    dataes.pop()
                dataes.append(info)
                if len(infoes) >= len_rnn_seq:
                    infoes.pop()
                infoes.append(info)
            else:  # 第一次执行预测，没有数据可输入网络，所以随机执行动作
                actions = random_action()
                control_by_vec(actions[0], duration=0.2, with_shift=False)
                time.sleep(0.1)
                state = 0
                x = getScreenPreprocess()  # 动作执行之后的图像
                data, info = retriveInputAgainstException()  # 动作执行之后的游戏数据
                xes.append(x)
                dataes.append(info)
                infoes.append(info)
            reward = getReward(data, previous_dataes)  # 本次动作的reward
            down = False
            speeds.append(data.car.Speed)  # 车辆速度
            wheel_speeds.append(data.car.WheelSpeed)  # 车轮速度
            if len(speeds) > num_step_to_check_trapped:  # 检车车辆是否控制
                max_speed = max(speeds)  # 最大速度
                mean_wheel_speed = sum(wheel_speeds) / \
                    len(wheel_speeds)  # 平均车轮速度
                if max_speed < 1 and mean_wheel_speed > 6:  # 空转(撞上车辆/墙壁等)
                    reward = -5  # 空转的reward负数
                    down = True  # 本回合失败
                speeds.clear()  # 清空列表
                wheel_speeds.clear()  # 清空列表
            # 车辆位置到起点终点直线的距离太大，表示偏离航线
            if getDisctanceFromPointToLine(data.startPosition, data.endPosition, data.car.Position) > 35:
                reward = -5  # 偏离航线负分
                down = True  # 本回合失败
            if distance(data.car.Position, data.endPosition) > 220:
                reward = -10  # 远离终点负分
                down = True  # 本回合失败
            if distance(data.car.Position, data.endPosition) < 20:
                reward = 25  # 到达终点高分
                down = True  # 回合成功
            reward = reward / 10
            if (len(dataes) >= len_rnn_seq) \
                    and (len(previous_dataes) >= len_rnn_seq) \
                    and (len(previous_xes) >= len_rnn_seq)\
                    and (len(xes) >= len_rnn_seq)\
                    and (len(infoes) >= len_rnn_seq)\
                    and (len(previous_infoes) >= len_rnn_seq):  # 序列的长度达到了可训练的标准
                record = Record()
                # 必须将tensor转换为 numpy保存，否则先前的计算图不会被清理，将导致内存溢出
                record.p_xes = [x.cpu().data.numpy() for x in previous_xes]
                record.p_infoes = [info.cpu().data.numpy()
                                   for info in previous_infoes]
                record.xes = [x.cpu().data.numpy() for x in xes]
                record.state = state.cpu().data.numpy()
                record.reward = reward
                record.infoes = [info.cpu().data.numpy() for info in infoes]
                record.down = down
                record.actions = actions.cpu().data.numpy() if isinstance(
                    actions, torch.Tensor) else actions
                logger.info("r: {};actions: {}; down: {}".format(
                    reward, record.actions, down))
                if len(buffer) >= max_buff_num:
                    buffer.pop()  # 去掉最旧的数据
                buffer.append(record)  # 加上最新的数据
                disToEnd = distance(
                    data.endPosition, data.car.Position)  # 到终点的距离
                print("reward at {} is {}, distance2end {}".format(
                    counter, reward, disToEnd))
            if num_step_from_last_down >= max_num_step_a_episode:
                down = True  # 跑了很多步也到不了终点，应当判失败
            if down:
                # 如果回合结束(成功或失败)
                backToStartPoint()  # 回到起点
                print("back to start at step {}".format(counter))
                logger.info("back to start at step {}".format(counter))
                time.sleep(1)
            if down and reward >= 10:  # 回合成功
                goodEpisode += 1
                print("aowsome agent finitshed the game")
                logger.info("aowsome agent finitshed the game")
                break
            if down and reward < 0:  # 回合失败
                badEposode += 1
                print("too bad agent crash the game")
                logger.info("too bad agent crash the game")
                break
            if down:
                num_step_from_last_down = 0
            if not down:
                num_step_from_last_down += 1

        # loss and learn

        if len(buffer) > least_buffer_for_learning:  # 如果buffer 的 长度达到学习的标准
            print("step {} learn".format(counter))
            down_index = -1  # 在buffer里面有没有回合结束的帧，如果没有就是 -1
            for i, record in enumerate(buffer):
                if record.down == True:
                    down_index = i
                    break
            if down_index != -1 and down_index < learn_batch_num:
                # 如果回合结束的帧，并在本次训练范围之内， 那么结束帧后面的数据不进入训练
                learn_batch_num = down_index
            rewards = []
            for i, record in enumerate(buffer[:learn_batch_num]):
              # 每次选取buffer 的前 learn_batch_num 个数据进行训练
                for j, next_record in enumerate(buffer[i+1: i+n]):
                  # 使用 discount 计算 target reward
                    if next_record.down == False:
                        reward += gamma**(j+1) * next_record.reward
                    else:
                        break
                rewards.append(reward)
            _p_xes = torch.cat([torch.tensor(np.array(x), dtype=dtype, device=device)
                                for record in buffer[:learn_batch_num] for x in record.p_xes], dim=0)  # shape (learn_batch_num,len_rnn_seq) => (learn_batch_num * len_rnn_seq,)
            _p_infoes = torch.cat([torch.tensor(np.array(info), dtype=dtype, device=device)
                                   for record in buffer[:learn_batch_num] for info in record.p_infoes], dim=0)  # shape (learn_batch_num,len_rnn_seq) => (learn_batch_num * len_rnn_seq,)
            _xes = torch.cat([torch.tensor(np.array(x), dtype=dtype, device=device)
                              for record in buffer[:learn_batch_num+n] for x in record.xes], dim=0)  # shape (learn_batch_num + n,len_rnn_seq) => ((learn_batch_num + n) * len_rnn_seq,)
            _infoes = torch.cat([torch.tensor(np.array(
                info), dtype=dtype, device=device) for record in buffer[:learn_batch_num+n] for info in record.infoes], dim=0)  # shape (learn_batch_num + n,len_rnn_seq) => ((learn_batch_num + n) * len_rnn_seq,)
            p_policise, p_states = net.forward(_p_xes, _p_infoes)
            _, states = net.forward(_xes, _infoes)
            _states = []
            for i, record in enumerate(buffer[: learn_batch_num]):
                state = states[i]
                for j, next_record in enumerate(buffer[i+1: i+n]):
                  # 使用 discount 计算 n 步 后的target state(值函数)
                    if next_record.down == False:
                        state += gamma**(j+1) * states[i+j]
                    else:
                        break
                _states.append(state)
            rewards = torch.tensor([[torch.tensor(reward, dtype=dtype, device=device)]
                                    for reward in rewards], dtype=dtype, device=device)
            states = torch.stack(_states)
            yes = rewards + states
            state_loss = torch.mean(torch.pow(p_states - yes, 2))
            advantages = - p_states
            max_policy, _ = torch.max(p_policise, 1)
            min_policy, _ = torch.min(p_policise, 1)
            smaller_than_0_index = (max_policy < 0)  # max_policy 中 < 0 的索引
            smaller_than_0_index = smaller_than_0_index.float()
            max_policy = max_policy - smaller_than_0_index * \
                min_policy  # 如果有 < 0，那么 max_policy -= 1 * min_policy
            policise_loss = torch.mean(
                torch.log(max_policy) * advantages)
            print("policy loss {}, state loss {}".format(
                policise_loss.cpu().data.numpy(), state_loss.cpu().data.numpy()))
            loss = state_loss + policise_loss  # 总的loss
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 梯度计算
            torch.nn.utils.clip_grad_norm_(train_parameters, 0.5)  # 梯度清理
            optimizer.step()  # 梯度下降

        # record
        counter += 1
        if counter % 100 == 0 and counter != 0:
            # 保存 state_dict
            torch.save(
                net.state_dict(), "./saved_model/AC_resnet_{}_state_dict.pt".format(counter))
            print("model saved in {} iterations".format(counter))
            logger.info("model saved in {} iterations".format(counter))


class Record:
    # buffer 的纪录类
    def __init__(self):
        self.p_xes = None
        # self.p_data = None
        self.reward = None
        self.actions = None
        self.state = None
        self.down = False
        self.xes = None
        self.p_infoes = None
        self.infoes = None
        # self.data = None


train()
