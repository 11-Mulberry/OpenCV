import cv2
import numpy as np

image_path = 'P2_No1.jpg'
image = cv2.imread(image_path)


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    # & 输出一个 rows * cols 的矩阵（imgArray）
    # print(rows, cols)
    # & 判断imgArray[0] 是不是一个list
    rowsAvailable = isinstance(imgArray[0], list)
    # & imgArray[][] 是什么意思呢？
    # & imgArray[0][0]就是指[0,0]的那个图片（我们把图片集分为二维矩阵，第一行、第一列的那个就是第一个图片）
    # & 而shape[1]就是width，shape[0]是height，shape[2]是
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    # & 例如，我们可以展示一下是什么含义
    # cv2.imshow("img", imgArray[0][1])

    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        # & 设置零矩阵
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image


resized_image = resize_image(image, 30)


def detect_color(image):
    # 定义颜色阈值范围
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([255, 50, 255])
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 30])

    # 将图像转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 创建黑白色彩区域的掩码
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    black_mask = cv2.inRange(hsv, black_lower, black_upper)

    # 如果白色掩码中有像素，则认为该矩形为白色
    if cv2.countNonZero(white_mask) > 0:
        return "White"
    # 如果黑色掩码中有像素，则认为该矩形为黑色
    elif cv2.countNonZero(black_mask) > 0:
        return "Black"
    else:
        return "Unknown"


cropped_regions = []
Cropped_Regions = []


def getContours(img):
    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area < 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            # print(len(approx))
            objCor = len(approx)

            if objCor < 5:
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
                center_x = (x + x + w) / 2
                center_y = (y + y + h) / 2
                center = (x + w // 2, y + h // 2)
                # print("Rectangle center:", (center_x, center_y))  # 输出中心点坐标
                rectangles.append((center_x, center_y))

                # 裁剪出矩形区域
                cropped_region = image[y:y+h, x:x+w]
                cropped_regions.append(cropped_region)
                Cropped_Regions.append((x, y, w, h))

                # 检测矩形区域的颜色
                color = detect_color(cropped_region)
                print("Color of the rectangle:", color)

    return rectangles


imgContour = resized_image.copy()
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 1)
canny = cv2.Canny(blur, 50, 50)

rectangles = getContours(canny)
for i, rectangle in enumerate(rectangles):
    print(f"Rectangle Center {i + 1}:", rectangle[0], rectangle[1])
# imgStack = stackImages(0.3, ([gray, blur, canny]))

# cv2.imshow('gray', gray)
cv2.imshow('imgContour', imgContour)

# 在循环结束后显示所有裁剪的区域
for i, region in enumerate(cropped_regions):
    cv2.imshow(f'cropped_region_{i}', region)

for rect in Cropped_Regions:
    x, y, w, h = rect
    print("x:", x, "y:", y, "w:", w, "h:", h)


cv2.waitKey(0)
cv2.destroyAllWindows()
