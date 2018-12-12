
import time
import sys
import signal

from capture import pictureCapture
from PyMata.pymata import PyMata
from pymata_aio.pymata3 import PyMata3
import cv2

FLAG = 1
# 超声波传感器初始化
board = PyMata("COM5", verbose=True)
board.sonar_config(12, 12)
# 舵机初始化
SERVO_MOTOR = 9
board.servo_config(SERVO_MOTOR)

# 视频初始化


# 杆子抬起放下操作
def arise_down():
    board.analog_write(SERVO_MOTOR, 0)
    time.sleep(3)
    board.analog_write(SERVO_MOTOR, 95)
    # time.sleep(5)
    print('车辆通行')
    # board.analog_write(SERVO_MOTOR, 0)


# 读取距离当距离小于一定值时,返回开始进行操作
# def getInstance():


# 剩下的工作时调用板子开始进行拍照, 并将视频保存下来
# 设置一定的延时进行启动
time.sleep(1)

# 主启动程序,判断程序在函数中
if __name__ == '__main__':
    print("系统成功启动！")
    cam = pictureCapture()

    while True:
        # 摄像头启动
        ret, cam.frame = cam.cap.read()
        cv2.imshow("capture", cam.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # cv2.imwrite("fangjian2.jpeg", frame)
            cam.captureImage()
            # cv2.waitKey(1)
            # self.cap.release()
            cv2.destroyAllWindows()
            continue
        data = board.get_sonar_data()
        distance = data[12][1]
        print(str(distance))
        if (distance < 30 and distance > 1):
            # board.close()
            isGo = cam.captureImage()
            if isGo:
                arise_down()
            else:
                print('车牌识别部分出现了问题')
        else:
            pass
        time.sleep(.2)
