import cv2
import numpy as np


def find_rectangles(image_path):
    # 读取图像并转换为灰度图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化图像
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        # area = cv2.contourArea(contour)
        # print(area)
        # if area > 500:
        #     cv2.drawContours(imgContour, contour, -1, (255, 0, 0), 3)
        #     peri = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.02*peri, True)
        #     print(len(approx))
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


# 测试函数
image_path = 'P2_No1.jpg'  # 修改为你的图像路径
rectangles = find_rectangles(image_path)
for i, rectangle in enumerate(rectangles):
    print(f"Rectangle {i + 1}:")
    print("Top left:", rectangle[0])
    print("Top right:", rectangle[1])
    print("Bottom right:", rectangle[2])
    print("Bottom left:", rectangle[3])
    print("Center:", rectangle[4])
    print()
