from PIL import ImageGrab
import cv2


def screen_grab(screen_pos):
    """截取屏幕上指定位置的图像

    Arguments:
      screen_pos {[list]} -- [[x1,y1],[x2,y2]]
      x1 -- 左上x
      y1 -- 左上y
      x2 -- 右下x
      y2 -- 右下y

    Returns:
      [PIL.image] -- 图片
    """

    left_x, left_y = screen_pos[0]
    right_x, right_y = screen_pos[1]
    image = ImageGrab.grab((left_x, left_y, right_x, right_y))
    return image
