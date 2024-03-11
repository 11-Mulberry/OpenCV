import cv2
import numpy as np

# def box(image):


def calibrate_image(image):
    # 假设已经获得空心方框和黑色实心方块的位置
    # 这里使用假数据代替
    # 实际应用中需要使用图像处理技术来检测这些位置
    # 空心方框和黑色实心方块的位置坐标
    hollow_box_position = (50, 50)  # 假设在图像的(50, 50)处
    black_box_position = (450, 450)  # 假设在图像的(450, 450)处

    # 计算旋转角度
    angle = np.arctan2(black_box_position[1] - hollow_box_position[1],
                       black_box_position[0] - hollow_box_position[0])
    angle = np.degrees(angle)

    # 旋转图像
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1)
    calibrated_image = cv2.warpAffine(image, M, (cols, rows))

    return calibrated_image


def detect_mines(image):
    # 假设地雷区域在标定后的图像的位置范围是 (100, 100) 到 (500, 500)
    # 实际应用中需要根据实际情况来确定地雷区域的位置
    mine_region = image[100:500, 100:500]

    # 使用霍夫圆变换检测圆形地雷
    gray = cv2.cvtColor(mine_region, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=20, param1=50, param2=30, minRadius=10, maxRadius=30)

    # 绘制矩形框和输出地雷位置
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # 圆心坐标
            center = (circle[0], circle[1])
            # 圆半径
            radius = circle[2]
            # 输出地雷的棋盘格坐标
            # 这里仅做示例，实际需要根据棋盘格的划分来确定坐标
            print("Mine position (x, y):", center)
            # 绘制矩形框
            x, y = center[0] - radius, center[1] - radius
            w, h = 2 * radius, 2 * radius
            cv2.rectangle(mine_region, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image


# 读取原始图片
original_image = cv2.imread('P2_No1.jpg')

# 校准图片位置
calibrated_image = calibrate_image(original_image)

# # 检测地雷
# mines_detected_image = detect_mines(calibrated_image)

# 保存标记好的图片
# cv2.imwrite('result/calibrated_and_detected_mines.jpg', mines_detected_image)
cv2.imshow(calibrated_image)
