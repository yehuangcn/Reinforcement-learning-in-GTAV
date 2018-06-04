class Vector:
    """向量
    """

    def __init__(self, vector=None):
        """从字典中获取 X Y Z 键的值

        Keyword Arguments:
            vector {dict} -- 字典向量 (default: {None})
        """

        if vector is None:
            self.X = None
            self.Y = None
            self.Z = None
        else:
            self.X = vector['X']
            self.Y = vector['Y']
            self.Z = vector['Z']


class Entity:
    """物体
    """

    def __init__(self, entity):
        """从字典中获取相应键的值构建Entity对象

        Arguments:
            entity {dict} -- 对象字典
        """

        self.IsOnScreen = entity['IsOnScreen']  # 物体是否在屏幕上
        self.Position = Vector(entity['Position'])  # 位置
        self.RightVector = Vector(entity['RightVector'])  # 右方向向量
        self.Rotation = Vector(entity['Rotation'])  # 旋转方向
        self.ForwardVector = Vector(entity['ForwardVector'])  # 前方向向量
        self.HeightAboveGround = entity['HeightAboveGround']  # 高出地面的距离
        self.Model = entity['Model']  # 物体的种类
        self.Velocity = Vector(entity['Velocity'])  # 速度向量


class Player:
    """游戏玩家
    """

    def __init__(self, player):
        self.IsHuman = player['IsHuman']  # 是否是人类
        self.IsOnFoot = player['IsOnFoot']  # 是否在站立的
        self.IsPlayer = player['IsPlayer']  # 是否是玩家本人
        self.IsOnScreen = player['IsOnScreen']  # 是否在屏幕上
        self.Position = Vector(player['Position'])  # 位置
        self.RightVector = Vector(player['RightVector'])  # 右方向向量
        self.Rotation = Vector(player['Rotation'])  # 旋转方向
        self.ForwardVector = Vector(player['ForwardVector'])  # 前方向向量
        self.Velocity = Vector(player['Velocity'])  # 速度向量


class Vehicle:
    def __init__(self, vehicle):
        self.IsOnScreen = vehicle['IsOnScreen']  # 是否在屏幕上
        self.Position = Vector(vehicle['Position'])  # 位置
        self.RightVector = Vector(vehicle['RightVector'])  # 车辆右方向
        self.Rotation = Vector(vehicle['Rotation'])  # 旋转方向
        self.ForwardVector = Vector(vehicle['ForwardVector'])  # 车辆前方向
        self.HeightAboveGround = vehicle['HeightAboveGround']  # 距离地面高度
        self.Velocity = Vector(vehicle['Velocity'])  # 速度向量
        self.RightHeadLightBroken = vehicle['RightHeadLightBroken']  # 前灯是否损坏
        self.LeftHeadLightBroken = vehicle['LeftHeadLightBroken']  # 尾灯是否损坏
        self.LightsOn = vehicle['LightsOn']  # 车灯是否打开
        self.EngineRunning = vehicle['EngineRunning']  # 引擎是否启动
        self.Health = vehicle['Health']  # 车辆的健康程度
        self.MaxHealth = vehicle['MaxHealth']  # 车辆的最大健康程度
        self.SearchLightOn = vehicle['SearchLightOn']  # 探照灯是否打开
        self.IsOnAllWheels = vehicle['IsOnAllWheels']  # ？

        # 是否在红绿灯前停下
        self.IsStoppedAtTrafficLights = vehicle['IsStoppedAtTrafficLights']
        self.IsStopped = vehicle['IsStopped']  # 车辆是否停下
        self.IsDriveable = vehicle['IsDriveable']  # 车辆是否是可以驾驶的
        self.IsConvertible = vehicle['IsConvertible']  # ？
        # 前保险杠是否损坏
        self.IsFrontBumperBrokenOff = vehicle['IsFrontBumperBrokenOff']
        # 后保险杠是否损坏
        self.IsRearBumperBrokenOff = vehicle['IsRearBumperBrokenOff']
        self.IsDamaged = vehicle['IsDamaged']  # 车辆是否有损坏
        self.Speed = vehicle['Speed']  # 速度
        self.BodyHealth = vehicle['BodyHealth']  # 车身健康程度
        self.MaxBraking = vehicle['MaxBraking']  # 最大刹车
        self.MaxTraction = vehicle['MaxTraction']  # 最大牵引
        self.EngineHealth = vehicle['EngineHealth']  # 引擎健康程度
        self.SteeringScale = vehicle['SteeringScale']  # 转向
        self.SteeringAngle = vehicle['SteeringAngle']  # 转向角
        self.WheelSpeed = vehicle['WheelSpeed']  # 轮胎转速
        self.Acceleration = vehicle['Acceleration']  # 加速 -1 1 0
        self.FuelLevel = vehicle['FuelLevel']  # 燃油数量(目测永远用不完)
        self.CurrentRPM = vehicle['CurrentRPM']  # 现在的每分钟转速
        self.CurrentGear = vehicle['CurrentGear']  # ？
        self.HighGear = vehicle['HighGear']  # ？


class Data:
    def __init__(self, data):
        # try:
        self.charactor = Player(data['charactor'])  # 玩家
        self.car = Vehicle(data['car'])  # 车辆
        self.endPosition = Vector(data['endPosition'])  # 起始点
        self.startPosition = Vector(data['startPosition'])  # 终点
        self.time_since_player_drove_against_traffic = data[
            'time_since_player_drove_against_traffic']  # 上次闯红灯的时间
        # 上次开上人行道距离现在的时间
        self.time_since_player_drove_on_pavement = data['time_since_player_drove_on_pavement']
        # 上次撞人距离现在的时间
        self.time_since_player_hit_ped = data['time_since_player_hit_ped']
        # 上次撞车距离现在的时间
        self.time_since_player_hit_vehicle = data['time_since_player_hit_vehicle']
        self.near_by_vehicles = [Vehicle(item)
                                 for item in data['near_by_vehicles']]  # 附近的车辆
        self.near_by_peds = [Player(item)
                             for item in data['near_by_peds']]  # 附近的行人
        self.near_by_props = [Entity(item)
                              for item in data['near_by_props']]  # 附近的物体（不包括地图物体）
        self.near_by_touching_peds = [
            Player(item) for item in data['near_by_touching_peds']]  # 附近有触碰的车辆
        self.near_by_touching_vehicles = [
            Vehicle(item) for item in data['near_by_touching_vehicles']]  # 附近有触碰的行人
        self.near_by_touching_props = [
            Entity(item) for item in data['near_by_touching_props']]  # 附近有触碰的物体
        self.next_position_on_street = Vector(
            data['next_position_on_street'])  # 街道的下一个位置
        self.forward_vector3 = Vector(data['forward_vector3'])  # 车头方向
        self.radius = data['radius']  # 附近的范围(半径)
        self.onRoad = data['onRoad']  # 是否在道路上(不包括人行道和绿化等)
        self.is_ped_injured = data['is_ped_injured']  # 玩家是否受伤
        self.is_ped_in_any_vehicle = data['is_ped_in_any_vehicle']  # 玩家是否在车里
        self.is_player_in_water = data['is_player_in_water']  # 车辆是否在水里
