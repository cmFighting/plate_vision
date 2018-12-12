# 有个关键点 需要注意,模型处理的是灰度图, 不是二值图像
import tensorflow as tf
import numpy as np
import cv2

# 通过索引将下面具体的值取出
chinese = ['川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀',
          '津', '京', '吉','辽', '鲁', '蒙', '闽', '宁', '青',
          '琼', '陕', '苏', '晋', '皖', '湘', '新', '豫',
          '渝', '粤', '云', '藏', '浙']

english = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

later = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def getResult(cjs):
    results = []
    sess1 = tf.InteractiveSession()  # 可以让我们使用先创建session,之后再进行图的创建
    # 去除第一个模型判断中文字符
    saver = tf.train.import_meta_graph('train_model/model/chinese/chinese.meta')
    saver.restore(sess1, 'train_model/model/chinese/chinese')
    graph = tf.get_default_graph()
    input_x = sess1.graph.get_tensor_by_name("Mul:0")
    y_conv2 = sess1.graph.get_tensor_by_name("final_result:0")
    #img = cv2.imread('new32.jpg', cv2.IMREAD_GRAYSCALE)
    x_img = np.reshape(cjs[0], [-1, 400])
    output = sess1.run(y_conv2, feed_dict={input_x: x_img})
    result = chinese[(np.argmax(output))]
    results.append(result)
    sess1.close()
    #print('the predict is %s' % chinese[(np.argmax(output))])

    # 取出第二个模型判断字母 这个暂时决定不用, 因为涉及到多次调用session
    # sess2 = tf.InteractiveSession()
    # saver = tf.train.import_meta_graph('train_model/model/english/english.meta')
    # saver.restore(sess2, 'train_model/model/english/english')
    # graph = tf.get_default_graph()
    # input_x = sess2.graph.get_tensor_by_name("Mul:0")
    # y_conv2 = sess2.graph.get_tensor_by_name("final_result:0")
    # # img = cv2.imread('new32.jpg', cv2.IMREAD_GRAYSCALE)
    # x_img = np.reshape(cjs[1], [-1, 400])
    # output = sess2.run(y_conv2, feed_dict={input_x: x_img})
    # result = english[(np.argmax(output))]
    # results.append(result)
    # #print('the predict is %s' % english[(np.argmax(output))])
    # sess2.close()

    # 取出第三个模型判断后面的5个字符 新的方法 要把图放置在合适的位置进行判断才可以, 图要进行重载才行
    graph3 = tf.Graph()
    with graph3.as_default():
        saver = tf.train.import_meta_graph('train_model/model/later/later.meta')
    sess3 = tf.Session(graph=graph3)
    with sess3.as_default():
        with graph3.as_default():
            saver.restore(sess3, 'train_model/model/later/later')# 从恢复点恢复参数
    input_x = sess3.graph.get_tensor_by_name("Mul:0")
    y_conv2 = sess3.graph.get_tensor_by_name("final_result:0")
    # img = cv2.imread('new32.jpg', cv2.IMREAD_GRAYSCALE)
    for cj in cjs[1:7]:
        x_img = np.reshape(cj, [-1, 400])
        output = sess3.run(y_conv2, feed_dict={input_x: x_img})
        result = later[(np.argmax(output))]
        results.append(result)
        #print('the predict is %s' % later[(np.argmax(output))])
    sess3.close()

    # sess3 = tf.InteractiveSession()
    # saver = tf.train.import_meta_graph('train_model/model/later/later.meta')
    # saver.restore(sess3, 'train_model/model/later/later')
    # graph = tf.get_default_graph()
    # input_x = sess3.graph.get_tensor_by_name("Mul:0")
    # y_conv2 = sess3.graph.get_tensor_by_name("final_result:0")
    # # img = cv2.imread('new32.jpg', cv2.IMREAD_GRAYSCALE)
    # for cj in cjs[1:7]:
    #     x_img = np.reshape(cj, [-1, 400])
    #     output = sess3.run(y_conv2, feed_dict={input_x: x_img})
    #     result = later[(np.argmax(output))]
    #     results.append(result)
    #     #print('the predict is %s' % later[(np.argmax(output))])
    # sess3.close()

    # print('done')
    return results
