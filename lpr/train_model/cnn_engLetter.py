import time
import numpy as np
import tensorflow as tf
import cv2
import os


# 权重初始化
# 定义权重,一个像素对应一个权重,不过这个会在后面的卷积核进行调整
# 因为一开始的权重是随机的,为了让这个随机更加合理一些,采用正太分布来进行合理的分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置量初始化,偏置量均初始化为以下的数据,就是shape大小的0.1的向量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 中间的两个数据是有用的,前后两个数据是默认的,帮助我们进行数据的一致性,如果采用不忽视边距的方法,可以让我们的输入和输出的大小是一致的
# 方便我们进行处理
# 卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 这个是基本的池化的操作,因为都是偶数的,所以这边这个不是重点,不需要进行调整
# 池化用简单传统的2x2大小的模板做max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 用ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。
# 然后每100次迭代输出一次日志
# 数据初始化
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
input_count = 0
for name in labels:
    dir = 'train/chars2/'+name
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            input_count += 1
print(input_count)
input_images = np.array([[0] * 400 for i in range(input_count)])
input_labels = np.array([[0] * 26 for i in range(input_count)])
index = 0
for name_index in range(0, len(labels)):
    #print(labels[name_index])
    dir = 'train/chars2/'+labels[name_index]
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            #print(filename)
            img = cv2.imread('train/chars2/'+labels[name_index]+'/'+filename, cv2.IMREAD_GRAYSCALE)
            #cv2.imshow('test', img)
            #cv2.waitKey(0)
            #ret, image_binary = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
            image = np.reshape(img, [-1, 400])
            #这边输入的值是0和255,其中255表示白色,0表示黑色
            #print(image)
            input_images[index] = image
            #每个对应有10个值,将符合要求的那个值设置为1
            input_labels[index][name_index] = 1
            index += 1
# ####这边就需要注意了,这个不是784了,应该是你图片的大小,这边的这个y也是实际你的label的大小
x = tf.placeholder(tf.float32, [None, 400],name='Mul')  # 图像输入向量
y_ = tf.placeholder("float", [None, 26])  # 实际分布


# 这个是没关系的,32个filter,通道数是1,因为是灰度图或者说是二值图像,
# 采用的是5*5的卷积核
# 第一层卷积由一个卷积接一个max pooling完成。
# 卷积在每个5x5的patch中算出32个特征。卷积的权重张量是[5, 5, 1, 32]，
# 前两个维度是patch的大小（5x5），接着是输入的通道数目（1），最后是输出的通道数目（32）。
W_conv1 = weight_variable([5, 5, 1, 32])

# 对于每一个输出通道都有一个对应的偏置量。故为32。
b_conv1 = bias_variable([32])


# ##这边需要注意,这是的图片转化应该是将图片转化为你输入的图片的实际大小
# 为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，
# 最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1, 20, 20, 1])

# 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积，把几个类似的层堆叠起来，构建一个更深的网络。
# 第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# #这边也需要注意,第一次池化,因为是2*2,所以减为1半是14*14,第二次池化也是减少一般所以是7*7
# 密集连接层
# 图片尺寸减小到7x7，加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([5 * 5 * 64, 1024])
b_fc1 = bias_variable([1024])

# 这边的道理是一样的,是让他们转化为对应维数的张量,-1表示的是缺省值
h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 64])
# 这边利用relu作为激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout为了减少过拟合而加入。
# 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

# 最后以softmax将他们对应到概率上,下面就是训练和评估模型了
# 第一个relu是全连接层之间转化用的,第二个全连接层是输出到对应的概率用的
# 之后还是需要看看具体的这些激活函数的意义以及这些激活函数的作用
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv2=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2,name="final_result")

# 训练和评估模型
# cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv + 1e-10))
# 明显采用这个模型对数据的训练可以更好
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# 启动session准备进行
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
batch_size = 60
iterations = 300
batches_count = int(input_count/batch_size)
remainder = input_count % batch_size
print('训练集分为'+str(batches_count+1)+'组,'+'前面每组'+str(batch_size)+'个数据'+'最后一组'+str(remainder)+'个数据')
tf.add_to_collection('output', y_conv2)
tf.add_to_collection('x', x)
# 对于一般的softmax函数来说,很可能会出现0的情况 这样就会导致你的模型无法进行使用,或者是训练效果不佳
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
        # 需要注意的是,这里的测试数据用的仅仅是原来的数据,所以其实准确率没有那么高
        iterate_accuracy = accuracy.eval(session=sess, feed_dict={x: input_images, y_: input_labels, keep_prob: 1.0})
        print('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
        if it >= iterations:
            break


# 将当前图设置为默认图
graph_def = tf.get_default_graph().as_graph_def()
# 将上面的变量转化成常量，保存模型为pb模型时需要,注意这里的final_result和前面的y_con2是同名，只有这样才会保存它，否则会报错，
# 如果需要保存其他tensor只需要让tensor的名字和这里保持一直即可
output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['final_result'])
saver = tf.train.Saver()
saver.save(sess, "model/english/")
sess.close()
print('英文字母模型训练完毕')




