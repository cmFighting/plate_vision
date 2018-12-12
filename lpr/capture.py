import cv2
import shortuuid
from test import getPlateResult
from process.DBopt import selectbyplate


class pictureCapture(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(1)

    def vedio(self):
        while True:
            ret, self.frame = self.cap.read()
            cv2.imshow("capture", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.imwrite("fangjian2.jpeg", frame)
                self.captureImage()
                # cv2.waitKey(1)
                # self.cap.release()
                cv2.destroyAllWindows()
                continue

    def captureImage(self):
        # 这个准确来说, 我应该到这个循环里面去调用, 通过接收一个信号,比如说键盘上的某一个值
        # 返回的数值直接传递回去到具体的操作函数中,触发验证等一系列的动作
        img_id = shortuuid.uuid()
        img_path ="cap_img/"+img_id+".jpg"
        print(img_path)
        cv2.imwrite(img_path, self.frame)
        cv2.destroyAllWindows()
        # img_path='cap_img/test11.jpg' # 暂时读取这张图片, 光线原因
        predict = getPlateResult(img_path)
        print("当前车辆:"+predict)
        isplate = selectbyplate(predict)
        if isplate:
            return True
        else:
            print('未找到该车辆信息!')
            return False
        # 继续让板子进行工作
        # 结果调用之后开始读取数据库, 将图片防止在数据库之中
        # 调用想数据库进行查询的方法, 最后返回true进行开始抬杆


# main函数
if __name__ == '__main__':
    pic = pictureCapture()
    pic.vedio()




