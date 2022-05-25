# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import pickle
# from dataset.mnist import load_mnist
from dataset_ch03.mnist import load_mnist
from common_all.functions import sigmoid, softmax
# 2022.4.6
# ychinata


# 3.6.2
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


# 3.6.2
def init_network():
    # pkl文件是学习到的参数
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


# 3.6.2
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = sigmoid(a3)
    return y


if __name__ == '__main__':
    # 3.6.2
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    cnt1 = 0
    cnt2 = 0
    for i in range(len(x)):
        cnt1 += 1 # 10000个
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]: # 预测正确
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

    # 3.6.3 批处理
    # 每100个进行一次批处理，共10000个，分成100批
    x, t = get_data()
    network = init_network()
    batch_size = 100
    # batch_size = 10000 # 1w个一批也行
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size): # step=100
        cnt2 += 1 # 100个
        x_batch = x[i:i+batch_size]  # shape:(100, 784)
        y_batch = predict(network, x_batch)  # shape:(100, 10)
        p = np.argmax(y_batch, axis=1)  # shape:(100, )
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))

