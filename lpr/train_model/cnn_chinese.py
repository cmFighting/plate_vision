import time
import numpy as np
import tensorflow as tf
import cv2
import os


# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置量初始化,偏置量均初始化为以下的数据,就是shape大小的0.1的向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。
# 然后每100次迭代输出一次日志
labels = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji',
          'zh_jin','zh_jing', 'zh_jI','zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing',
          'zh_qiong', 'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu',
          'zh_yu1','zh_yue','zh_yun','zh_zang','zh_zhe']
input_count = 0
for name in labels:
    dir = 'train/charsChinese/charsChinese/'+name
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1
print(input_count)
input_images = np.array([[0] * 400 for i in range(input_count)])
input_labels = np.array([[0] * 31 for i in range(input_count)])
index = 0
for name_index in range(0, len(labels)):
    #print(labels[name_index])
    dir = 'train/charsChinese/charsChinese/'+labels[name_index]
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            img = cv2.imread('train/charsChinese/charsChinese/'+labels[name_index]+'/'+filename, cv2.IMREAD_GRAYSCALE)
            image = np.reshape(img, [-1, 400])
            input_images[index] = image
            input_labels[index][name_index] = 1
            index += 1
x = tf.placeholder(tf.float32, [None, 400], name='Mul')  # 图像输入向量
y_ = tf.placeholder("float", [None, 31])  # 实际分布


W_conv1 = weight_variable([5, 5, 1, 32])


b_conv1 = bias_variable([32])


x_image = tf.reshape(x, [-1, 20, 20, 1])


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


W_fc1 = weight_variable([5 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 31])
b_fc2 = bias_variable([31])


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv2=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="final_result")


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
batch_size = 60
iterations = 500  # 为了让模型足够好,可以让他多跑几次
batches_count = int(input_count/batch_size)
remainder = input_count % batch_size
print('训练集分为'+str(batches_count+1)+'组,'+'前面每组'+str(batch_size)+'个数据'+'最后一组'+str(remainder)+'个数据')
#tf.add_to_collection('res', y_conv)
tf.add_to_collection('output', y_conv2)
tf.add_to_collection('x', x)
for it in range(iterations):
    # 这里的关键是要把输入数组转为np.array
    for n in range(batches_count):
        train_step.run(session=sess, feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size],
                                  y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
    if remainder > 0:
        start_index = batches_count * batch_size
        # 训练的时候,仅仅保持0.5的保持率
        train_step.run(session=sess, feed_dict={x: input_images[start_index:input_count - 1],
                                  y_: input_labels[start_index:input_count - 1], keep_prob: 0.5})

    # 每完成50次迭代，判断准确度是否已达到100%，达到则退出迭代循环
    iterate_accuracy = 0
    if it % 50 == 0:
        iterate_accuracy = accuracy.eval(session=sess, feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
        print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
        if it >= iterations:
            break


# 将当前图设置为默认图
graph_def = tf.get_default_graph().as_graph_def()
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['final_result'])
saver = tf.train.Saver()
saver.save(sess, "model/chinese/test")
sess.close()
print('中文字符训练结束')




