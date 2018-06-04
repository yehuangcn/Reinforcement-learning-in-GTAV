screen_width = 800  # 截屏宽度
screen_height = 600  # 截屏高度
leftx = 0  # 左上角x坐标
lefty = 20  # 左上角y坐标
screen_pos = [[leftx, lefty], [leftx + screen_width,
                               lefty + screen_height]]  # 截屏位置，以屏幕左上角为原点
channel = 3  # 图片通道数

samll_map_leftx = leftx + 5  # 小地图左上角x坐标
samll_map_lefty = lefty + 465  # 小地图左上角y坐标
small_map_pos = [[samll_map_leftx, samll_map_lefty],
                 [samll_map_leftx + 160, samll_map_lefty + 115]]  # 小地图截屏位置，以屏幕左上角为原点
