import cv2
import numpy as np

BLUR = 3
MORPHOLOGYR = 5
MORPHOLOGYC = 12
Min_Area = 2000  # 车牌区域允许最大面积
minPlateRatio = 0.5  # 车牌最小比例
maxPlateRatio = 6  # 车牌最大比例
lower_blue = np.array([100, 40, 50])  # 车牌蓝色区间比例
higher_blue = np.array([140, 255, 255])


def new_location(filename):
    src_img = cv2.imread(filename)
    # cv2.imshow('1', src_img)
    # cv2.waitKey(0)
    pic_hight, pic_width = src_img.shape[:2]
    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower_blue, higher_blue)
    res = cv2.bitwise_and(src_img, src_img, mask=mask)
    # img = cv2.GaussianBlur(src_img, (BLUR, BLUR), 0)
    img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (BLUR, BLUR), 0)



    # 对图像进行开闭运算, 方便后期寻找轮廓
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    kernel = np.ones((MORPHOLOGYR, MORPHOLOGYC), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Min_Area]

    car_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # print(wh_ratio)
        # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 5.5:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            oldimg = cv2.drawContours(src_img, [box], 0, (0, 0, 255), 2)
            cv2.imshow("edge4", oldimg)
            cv2.waitKey(0)
        # print(rect)
    print(len(car_contours))

    card_imgs = []
    # 矩形区域可能是倾斜的矩形，需要矫正
    for rect in car_contours:
        # cv2.imshow("edge5", rect)
        if rect[2] > -1 and rect[2] < 1:  # 创造角度，使得左、高、右、低拿到正确的值
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除

        box = cv2.boxPoints(rect)
        heigth_point = right_point = [0, 0]
        left_point = low_point = [pic_width, pic_hight]
        for point in box:
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(src_img, M, (pic_width, pic_hight))
            # cv2.imshow("edge5", dst)
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            card_imgs.append(card_img)
        # cv2.imshow("card", card_img)
        # cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  # 负角度

            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(src_img, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            card_imgs.append(card_img)
        # cv2.imshow("cardx", card_img)
        # cv2.waitKey(0)

    for card_img in card_imgs:
        cv2.imshow('1', card_imgs[0])
        # cv2.imwrite('location.jpg',card_imgs[0])
        cv2.waitKey(0)

    return card_imgs[0]



def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


#     region = []
#     list_rate = []
#
#     car_contours = []
#     for i in range(len(contours)):
#         cnt = contours[i]
#         area = cv2.contourArea(cnt)
#         if area < 1000:
#             continue
#         rect = cv2.minAreaRect(cnt)
#         box = np.int32(cv2.boxPoints(rect))
#         height = abs(box[0][1] - box[2][1])
#         width = abs(box[0][0] - box[2][0])
#         ratio = float(width) / float(height)
#         rate = getxyRate(cnt)
#         # print("area", area, "ratio:", ratio, "rate:", rate)
#         if ratio > maxPlateRatio or ratio < minPlateRatio:
#             continue
#         region.append(box)
#         list_rate.append(ratio)
#     index = getSatifyestBox(list_rate)
#     region = region[index]
#     print(region)
#
#     Xs = [i[0] for i in region]
#     Ys = [i[1] for i in region]
#     x1 = min(Xs)
#     x2 = max(Xs)
#     y1 = min(Ys)
#     y2 = max(Ys)
#     hight = y2 - y1
#     width = x2 - x1
#     crop_img = src_img[y1:y1 + hight, x1:x1 + width]
#     # cv2.imshow('crop_img_1', crop_img)
#     # cv2.imwrite('crop_1.jpg', crop_img)
#     imgx = src_img.copy()
#     cv2.drawContours(src_img, [region], 0, (0, 255, 0), 2)
#     cv2.imshow("车牌定位",src_img)
#     cv2.waitKey(0)
#     return crop_img
#     # for cnt in contours:
#     #     rect = cv2.minAreaRect(cnt)
#     #     area_width, area_height = rect[1]
#     #     if area_width < area_height:
#     #         area_width, area_height = area_height, area_width
#     #     wh_ratio = area_width / area_height
#     #     # print(wh_ratio)
#     #     # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
#     #     if wh_ratio > 2 and wh_ratio < 5.5:
#     #         car_contours.append(rect)
#     #         box = cv2.boxPoints(rect)
#     #         box = np.int0(box)
#     #         oldimg = cv2.drawContours(src_img, [box], 0, (0, 0, 255), 2)
#     #         cv2.imshow("edge4", oldimg)
#     #         cv2.waitKey(0)
#     #     print(len(car_contours))
# # 对轮廓的高和宽求比例, 判断是否符合车牌
# def getxyRate(cnt):
#     x_height = 0
#     y_height = 0
#     x_list = []
#     y_list = []
#     for location_value in cnt:
#         location = location_value[0]
#         x_list.append(location[0])
#         y_list.append(location[1])
#     x_height = max(x_list) - min(x_list)
#     y_height = max(y_list) - min(y_list)
#     return x_height * (1.0) / y_height * (1.0)
#
# def getSatifyestBox(list_rate):
#     for index, key in enumerate(list_rate):
#         list_rate[index] = abs(key - 3)
#     #print(list_rate)
#     index = list_rate.index(min(list_rate))
#     #print(index)
#     return index

if __name__ == '__main__':
    new_location("D:/coding/codeOfPy/Antetokounmpo/timg/timg (9).jpg")
