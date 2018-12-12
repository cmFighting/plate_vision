import time
import sys
import signal

from PyMata.pymata import PyMata


# Ping callback function
# 这个函数的作用这是将函数打印出来




# Create a PyMata instance
board = PyMata("COM5", verbose=True)


def cb_ping(data):
    # time.sleep(2)
    print(str(data[2]) + ' centimeters')
    # if data[2] < 5:
    #     print('检测到物体')
    # else:
    #     print('no')



# you may need to press control-c twice to exit 使用两次contloc来进行程序的退出
def signal_handler(sig, frame):
    print('You pressed Ctrl+C!!!!')
    if board is not None:
        board.reset()
        board.close()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

# Configure Trigger and Echo pins
# Using default of 50 ms ping interval and max distance of 200 cm.
data = board.sonar_config(12, 12, cb_ping)


# Example of changing ping interval to 100, and max distance to 150
# board.sonar_config(12, 12, cb_ping, 100, 150)

time.sleep(20)
board.close()