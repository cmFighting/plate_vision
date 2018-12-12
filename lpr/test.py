from process import licensePlateLocation as li,new_split
from process.test import new_location
from train_model import readmodel
import cv2


def getPlateResult(filename):
    plate = li.location(filename)
    # cv2.imshow('a', plate)
    # cv2.waitKey(0)
    cjs = new_split.new_split(plate)
    # for cj in cjs:
    #     cv2.imshow("", cj)
    #     cv2.waitKey(0)
    results = readmodel.getResult(cjs)
    str_result = ''.join(results)
    return str_result


def getPlateResult_x(filename):
    # plate = li.location(filename)
    plate = new_location.new_location(filename)
    cv2.imshow("1", plate)
    cv2.waitKey(0)
    cjs = new_split.new_split(plate)
    results = readmodel.getResult(cjs)
    str_result = ''.join(results)
    return str_result


if __name__ == '__main__':
    nice = getPlateResult('cap_img/test11.jpg')
    print(nice)

