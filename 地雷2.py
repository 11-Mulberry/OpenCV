import cv2
import numpy as np


def rotate_image(image):
    # 获取空心方块和黑色方块的位置坐标
    # 这里使用假数据代替
    # 实际应用中需要使用图像处理技术来检测这些位置
    hollow_box_position = (100, 100)  # 假设在图像的(100, 100)处
    black_box_position = (500, 500)  # 假设在图像的(500, 500)处

    # 判断是否需要旋转图片
    if hollow_box_position[0] > black_box_position[0] and hollow_box_position[1] < black_box_position[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif hollow_box_position[0] < black_box_position[0] and hollow_box_position[1] > black_box_position[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif hollow_box_position[0] > black_box_position[0] and hollow_box_position[1] > black_box_position[1]:
        image = cv2.rotate(image, cv2.ROTATE_180)

    return image


def detect_mines(image):
    # 使用霍夫圆变换检测空心圆环
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=20, param1=50, param2=30, minRadius=10, maxRadius=30)

    # 绘制矩形框并输出空心圆环圆心的坐标
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # 圆心坐标
            center = (circle[0], circle[1])
            # 绘制矩形框
            x, y = circle[0] - circle[2], circle[1] - circle[2]
            w, h = 2 * circle[2], 2 * circle[2]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 输出圆心的棋盘格坐标
            region_x = int(center[0] / (image.shape[1] / 6)) + 1
            region_y = int(center[1] / (image.shape[0] / 4)) + 1
            print("Mine position (x, y):", (region_x, region_y))

    return image


# 读取原始图片
original_image = cv2.imread('original_image.jpg')

# 校准图片位置
calibrated_image = rotate_image(original_image)

# 检测地雷
mines_detected_image = detect_mines(calibrated_image)

# 保存标记好的图片
cv2.imwrite('result/calibrated_and_detected_mines.jpg', mines_detected_image)
