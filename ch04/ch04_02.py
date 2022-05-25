# coding: utf-8
# 2022.4.6
# ychinata
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
# from dataset.mnist import load_mnist
from dataset_all.mnist import load_mnist
from common_all.functions import sigmoid, softmax

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# 平均交差熵误差
def cross_entropy_error(y, t):
    # y是神经网络的输出，t是监督数据
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size
    # return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


def get_mnist_test_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def get_mnist_train_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_train, t_train


if __name__ == '__main__':
    t = [0,0,1,0,0,0,0,0,0,0]
    # example 1
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    mean_squared_error(np.array(y), np.array((t)))
    cross_entropy_error(np.array(y), np.array((t)))

    # 4.2.3
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
