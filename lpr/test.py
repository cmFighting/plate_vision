from process import characterSegmentation as ch, licensePlateLocation as li
from train_model import readmodel
import cv2

# cjs = li.split('crop.jpg')
# results = readmodel.getResult(cjs)
# print(results)
# name = ['京', 'A', '8', '0', '2', '9', '3']
# strx = ''.join(name)
# print(strx)
#print(cjs)


def getPlateResult(filename):
    plate = li.location(filename)
    cv2.imshow('a', plate)
    cv2.waitKey(0)

    cjs = ch.split(plate)
    results = readmodel.getResult(cjs)
    str_result = ''.join(results)
    return str_result

if __name__ == '__main__':
    nice = getPlateResult('test.jpg')
    print(nice)