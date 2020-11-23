import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# K折交叉验证
from sklearn.model_selection import KFold #train_test_split 一次划分，不再改变
#训练结果保存
import sys
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger("结果.txt")  # 保存到D盘

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('原始数据集：{0}{1}'.format(x_train.shape, x_test.shape))

x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)  # 28*28-》32*32
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)  # 28*28-》32*32
print('用0填充四边数据集：{0}{1}'.format(x_train.shape, x_test.shape))

# 数据集格式转换
x_train = x_train.astype('float32')
x_train = x_train.astype('float32')

x_train = x_train / 255  # 归一化
x_test = x_test / 255  # 归一化
print('归一化后数据集：{0}{1}'.format(x_train.shape, x_test.shape))

x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
print('整体归一化后数据集：{0}{1}'.format(x_train.shape, x_test.shape))

# 模型实例化
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu,
                           input_shape=(32, 32, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu,
                           input_shape=(32, 32, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=120, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=84, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax),
])
model.summary()
# 第二部分 模型训练
num_epochs = 11  # 训练次数
batch_size = 64  # 每个批次喂多少张图片
lr = 0.001  # 学习率

# 优化器
adam_optimizer = tf.keras.optimizers.Adam(lr)
loss_function = tf.keras.losses.sparse_categorical_crossentropy
model.compile(
    optimizer=adam_optimizer,
    loss=loss_function,
    metrics=['accuracy'])

import datetime


# 交叉验证
KF = KFold(n_splits=10,shuffle=False,random_state=100)

for train_index,dev_index in KF.split(x_train):

    X_train = x_train[train_index]
    dev_train = x_train[dev_index]

    start_time = datetime.datetime.now()
    x=model.fit(
        x=X_train,
        y=y_train[train_index],
        batch_size=batch_size,
        epochs=num_epochs)
    end_time = datetime.datetime.now()
    time_cost = end_time - start_time
    print('time_cost: ', time_cost)
    loss,accuracy = model.evaluate(dev_train, y_train[dev_index])
    print('模型交叉验证',loss,accuracy)

print('保存完成')




# 保存模型,需要自己建立文件夹
# model.save_weights('./check_points/my_check_point')
model.save('./model/leNet_model.h5')


'''image_index = 0

# 预测
pred = model.predict(x_test[image_index].reshape(1, 32, 32, 1))
print(pred.argmax())

# 显示
plt.imshow(x_test[image_index].reshape(32, 32))
cv2.imwrite('7.jpg',x_test[image_index].reshape(32, 32))
plt.show()'''