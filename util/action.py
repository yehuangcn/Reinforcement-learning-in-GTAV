import numpy as np
import random
from constant.key import Keys
from .key import PressKey, ReleaseKey
from util import list_equal
import cv2
import numpy as np
import win32api
import win32con
import time



# 可执行的动作列表
import win32api
action_list = ['w', 'wa', 'wd', 'sa', 'sd', "s"]


def vec2keys(vec):
    """向量转换为按键

    Arguments:
        vec {list 或者 np.array} -- 长度 为 6

    Returns:
        string  -- action_list 的 中的元素
    """

    assert len(vec) == 6
    index = np.argmax(vec)
    return action_list[index]

# 可读取的键的列表
key_list = "\bABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\"


def getKey():
    """获取按键

    Returns:
        list -- 按键列表
    """

    keys = []
    for key in key_list:
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys


def press_key(key, duration=None):
    """按下按键

    Arguments:
        key {string} -- 按键

    Keyword Arguments:
        duration {float} -- 按键持续时间 (default: {None})
    """

    PressKey(key)
    if duration is not None:
        time.sleep(duration)
        ReleaseKey(key)


def control(key, duration):
    """游戏动作，前/后/左前/左后/右前/右后

    Arguments:
        key {string} -- 按键
        duration {float} -- 按键时常
    """

    if 'w' == key:
        ReleaseKey(Keys.s)
        ReleaseKey(Keys.a)
        ReleaseKey(Keys.d)
        press_key(Keys.w)
    elif 'wa' == key:
        ReleaseKey(Keys.s)
        ReleaseKey(Keys.d)
        press_key(Keys.w)
        press_key(Keys.a, duration=duration)
    elif 'wd' == key:
        ReleaseKey(Keys.s)
        ReleaseKey(Keys.a)
        press_key(Keys.w)
        press_key(Keys.d, duration=duration)
    elif 'sa' == key:
        ReleaseKey(Keys.w)
        ReleaseKey(Keys.d)
        press_key(Keys.s)
        press_key(Keys.a, duration=duration)
    elif 'sd' == key:
        ReleaseKey(Keys.w)
        ReleaseKey(Keys.a)
        press_key(Keys.s)
        press_key(Keys.d, duration=duration)
    elif 's' == key:
        ReleaseKey(Keys.w)
        ReleaseKey(Keys.a)
        ReleaseKey(Keys.d)
        press_key(Keys.s)
    else:
        print("get wrong action keys : {}".format(keys))


def control_by_vec(vec, duration):
    """使用向量执行游戏动作

    Arguments:
        vec {np.array or list} -- 向量
        duration {float} -- 按键时常

    """

    keys = vec2keys(vec)
    control(keys, duration)


def random_action():
    """随机选择一个动作

    Returns:
        list -- 例如 [[0,0,0,1,0,0]]
    """

    action_index = random.randint(0, 5)
    temp_actions = np.zeros((1, 6))
    temp_actions[0][action_index] = 1
    return temp_actions
