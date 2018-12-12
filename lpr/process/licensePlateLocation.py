import os
import cv2
import numpy as np


minPlateRatio = 0.5  # 车牌最小比例
maxPlateRatio = 6  # 车牌最大比例
lower_blue = np.array([100, 40, 50])  # 车牌蓝色区间比例
higher_blue = np.array([140, 255, 255])


# 寻找符合车牌形状的矩形, 并添加到候选列表集合中
def findPlateNumberRegion(img):
    region = []
    contours_img, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("contours lenth is :%s" % (len(contours)))
    list_rate = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        rect = cv2.minAreaRect(cnt)
        box = np.int32(cv2.boxPoints(rect))
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)
        rate = getxyRate(cnt)
        print("area", area, "ratio:", ratio, "rate:", rate)
        if ratio > maxPlateRatio or ratio < minPlateRatio:
            continue
        region.append(box)
        list_rate.append(ratio)
    index = getSatifyestBox(list_rate)
    return region[index]


# 寻找最有可能是车牌的位置
def getSatifyestBox(list_rate):
    for index, key in enumerate(list_rate):
        list_rate[index] = abs(key - 3)
    #print(list_rate)
    index = list_rate.index(min(list_rate))
    #print(index)
    return index


# 对轮廓的高和宽求比例, 判断是否符合车牌
def getxyRate(cnt):
    x_height = 0
    y_height = 0
    x_list = []
    y_list = []
    for location_value in cnt:
        location = location_value[0]
        x_list.append(location[0])
        y_list.append(location[1])
    x_height = max(x_list) - min(x_list)
    y_height = max(y_list) - min(y_list)
    return x_height * (1.0) / y_height * (1.0)


# 传入当前图片的位置, 进行定位,返回最终的区域
def location(filename):
    img = cv2.imread(filename)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #cv2.imshow("hsv_img", hsv_img)

    mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("res", res)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)

    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    #cv2.imshow("gaussian", gaussian)

    sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
    #cv2.imshow("sobel", sobel)

    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow("binary", binary)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
    #cv2.imshow("closed", closed)

    # 目前下面这步需要进行确定, 可能是直接切割, 也可能是在灰度图像或者是二值图像上进行切割
    # 利用坐标信息直接进行提取
    region = findPlateNumberRegion(closed)
    Xs = [i[0] for i in region]
    Ys = [i[1] for i in region]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = img[y1:y1 + hight, x1:x1 + width]
    # cv2.imshow('crop_img_1', crop_img)
    # cv2.imwrite('crop_1.jpg', crop_img)
    #cv2.drawContours(img, [region], 0, (0, 255, 0), 2)
    # cv2.imshow("img", img)
    return crop_img


if __name__ == '__main__':
    file = "test.jpg"
    location(file)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
