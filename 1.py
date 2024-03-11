import cv2
import numpy as np


def find_rectangles(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        # 获取轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # 筛选出面积小于画面20%的矩形
        if area < 0.2 * image.shape[0] * image.shape[1]:
            # 获取矩形的四个顶点坐标及中心坐标
            top_left = (x, y)
            top_right = (x + w, y)
            bottom_right = (x + w, y + h)
            bottom_left = (x, y + h)
            center = (x + w // 2, y + h // 2)

            rectangles.append(
                (top_left, top_right, bottom_right, bottom_left, center))

    return rectangles

# 放缩函数


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


# 测试函数
image_path = 'P2_No1.jpg'  # 修改为你的图像路径

# 读取原始图像
image = cv2.imread(image_path)

# 放缩图像至50%
resized_image = resize_image(image, 50)

# 查找矩形
rectangles = find_rectangles(resized_image)
for i, rectangle in enumerate(rectangles):
    print(f"Rectangle {i + 1}:")
    print("Top left:", rectangle[0])
    print("Top right:", rectangle[1])
    print("Bottom right:", rectangle[2])
    print("Bottom left:", rectangle[3])
    print("Center:", rectangle[4])
    print()

# 绘制矩形
for rectangle in rectangles:
    cv2.rectangle(resized_image, rectangle[0], rectangle[2], (0, 255, 0), 2)

# 显示图像
cv2.imshow('Resized Rectangles', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
