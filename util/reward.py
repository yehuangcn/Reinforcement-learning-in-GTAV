from util.model import Data, Vector
import numpy as np
import math
from constant.log import logger
import json


def calReward(data):
    """根据Data计算reward
        disToEnd  -- 到终点的距离
        angle --  车头方向 和 终点方向的夹角
        data.onRoad -- 车辆是否在路上
        against_traffic -- 是否逆行
        drove_on_pavement -- 是否行驶上人行道
        hit_ped -- 是否撞了人
        hit_vehicle -- 是否撞了车
        abs(data.car.Speed - 30) -- 和理想速度的差距
        data.is_player_in_water -- 是否在水里
        num_near_by_vehicle -- 附近车辆的数量
        len(data.near_by_touching_peds) -- 附近碰撞的行人的数量
        len(data.near_by_touching_props) -- 附近碰撞物体的数量
        len(data.near_by_touching_vehicles) -- 附近碰撞的车辆的数量
        diff_Speed -- 移动速度和车轮速度的插件

    Arguments:
        data {Data} -- 数据

    Returns:
        [float] -- reward
    """

    assert isinstance(data, Data)
    time_since = 150  # ms
    against_traffic = data.time_since_player_drove_against_traffic < time_since and data.time_since_player_drove_against_traffic >= 0
    drove_on_pavement = data.time_since_player_drove_on_pavement < time_since and data.time_since_player_drove_on_pavement >= 0
    hit_ped = data.time_since_player_hit_ped < time_since and data.time_since_player_hit_ped >= 0
    hit_vehicle = data.time_since_player_hit_vehicle < time_since and data.time_since_player_hit_vehicle >= 0
    num_near_by_vehicle = len(data.near_by_vehicles)
    num_near_by_peds = len(data.near_by_peds)
    target_direction = minus(data.car.Position, data.endPosition)
    angle = includedAngle(target_direction, data.forward_vector3)
    + num_near_by_peds * 4
    disToEnd = distance(data.endPosition, data.car.Position)
    diff_Speed = abs(data.car.Speed - data.car.WheelSpeed)
    reward = - disToEnd * 4 \
        - angle * 10 \
        + data.onRoad * 20\
        - against_traffic * 8 \
        - drove_on_pavement * 8\
        - hit_ped * 6\
        - hit_vehicle * 6\
        - abs(data.car.Speed - 30)\
        - data.is_player_in_water * 20 \
        + num_near_by_vehicle * 4\
        - len(data.near_by_touching_peds) * 4\
        - len(data.near_by_touching_props) * 2  \
        - diff_Speed * 8\
        - len(data.near_by_touching_vehicles) * 2\
        + 800  # 使得reward为正值

    if reward < 1:
        reward = 1
    return reward / 200


def minus(vec1, vec2):
    """向量相减 vec1 - vec2

    Arguments:
        vec1 {Vector} -- 向量1
        vec2 {Vector} -- 向量2

    Returns:
        Vector -- 结果
    """

    assert isinstance(vec1, Vector)
    assert isinstance(vec2, Vector)
    vec = Vector()
    vec.X = vec2.X - vec1.X
    vec.Y = vec2.Y - vec1.Y
    vec.Z = vec2.Z - vec1.Z
    return vec


def multiply(vec1, vec2):
    """向量点乘

    Arguments:
        vec1 {Vector} -- 向量1
        vec2 {Vector} -- 向量2

    Returns:
        float -- 结果
    """

    return vec1.X * vec2.X + vec1.Y * vec2.Y + vec1.Z * vec2.Z


def nearbyVectors(from_vec, tos, limit):
    """处理几个nearby开头的向量列表，按照距离远近返回指定长度的列表，多则去尾， 少则补零

    Arguments:
        from_vec {Vector} -- 起始点
        tos {list[Vector]} -- 终点列表
        limit {int} -- 返回的列表的长度

    Returns:
        {list[Vector]} -- 以 from_vec 为原点的 长度为 limit 的Vector列表
    """

    tos.sort(key=lambda to_vec: distance(to_vec, from_vec))  # 从近到远排序
    if(len(tos) >= limit):  # 多则去尾
        return [minus(from_vec, to_vec) for to_vec in tos[:limit]]
    else:  # 少则补零
        direction_vecs = [minus(from_vec, to_vec) for to_vec in tos]
        direction_vecs.extend(
            [Vector({'X': 0, 'Y': 0, 'Z': 0}) for i in range(0, limit - len(tos))])
        return direction_vecs


def vector2Numpy(vec):
    """Vector对象转换为np.array,按照x,y,z的顺序排列

    Arguments:
        vec {Vector} -- 向量

    Returns:
        np.array -- [x,y,z]
    """

    return np.array([vec.X, vec.Y, vec.Z])


def vectors2Numpy(vecs):
    """Vector对象列表转换为np.array

    Arguments:
        vecs {list[Vector]} -- 向量列表

    Returns:
        np.array -- [[x,y,z]...]
    """

    return np.array([vector2Numpy(vec) for vec in vecs])


def includedAngle(vec1, vec2):
    """平面向量夹角，只看xy平面的夹角 

    Arguments:
        vec1 {Vector} -- 向量1
        vec2 {Vector} -- 向量2

    Returns:
        [float] -- 夹角的cos值
    """

    assert isinstance(vec1, Vector)
    assert isinstance(vec2, Vector)
    num_vec1 = np.array([vec1.X, vec1.Y])  # [x,y,z]只需要 [x,y]
    num_vec2 = np.array([vec2.X, vec2.Y])
    cos = np.dot(num_vec2, num_vec2) / \
        (np.linalg.norm(num_vec2)*(np.linalg.norm(num_vec1)))  # wrong
    cos = np.clip(cos, -1, 1)
    return cos


def distance(vec1, vec2):
    """两个向量的距离

    Arguments:
        vec1 {Vector} -- 向量1
        vec2 {Vector} -- 向量2

    Returns:
        [float] -- 向量的距离
    """

    assert isinstance(vec1, Vector)
    assert isinstance(vec2, Vector)
    num_vec1 = np.array([vec1.X, vec1.Y, vec1.Z])
    num_vec2 = np.array([vec2.X, vec2.Y, vec2.Z])
    return np.sqrt(np.sum(np.square(num_vec1 - num_vec2)))
