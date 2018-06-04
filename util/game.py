import math
import json
import requests
import sys
import time
sys.path.append('')
from util.key import PressKey, ReleaseKey
from util import count_down
from constant.key import Keys


def gameData(fromNet=False):
    """获取游戏数据
    如果 fromNet=False 那么从本地的 sample.json 获取，仅作测试用
    Keyword Arguments:
        fromNet {bool} -- 从网络获取 (default: {False})

    Returns:
        dict -- 数据
    """

    if fromNet:
        url = "http://localhost:31730/data"
        response = requests.get(url)
        return json.loads(response.content)
    else:
        with open("./util/sample.json") as f:
            data = json.load(f)
            return data


def backToStartPoint():
    """回到起点
    """

    PressKey(Keys.h)
    print("back")
    ReleaseKey(Keys.h)


def goToEndPoint():
    """前往终点(deprecated)
    """
    PressKey(Keys.lcontrol)
    PressKey(Keys.q)
    ReleaseKey(Keys.lcontrol)
    ReleaseKey(Keys.q)


def getDisctanceFromPointToLine(point1, point2, point0):
    """点0 到 点1 和点2 所在直线的距离

    Arguments:
        point1 {Vector} -- 点1
        point2 {Vector} -- 点2
        point0 {Vector} -- 点0

    Returns:
        [float] -- 距离
    """

    A = point2.Y - point1.Y
    B = point1.X - point2.X
    C = point2.X * point1.Y - point1.X * point2.Y
    distance = math.fabs(A * point0.X + B*point0.Y + C) / \
        math.sqrt(math.pow(A, 2) + math.pow(B, 2))
    return distance
