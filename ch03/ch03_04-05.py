# 2022.4.2
# ychinata
# chapter 3: neural network

# import pygame
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    y = x > 0
    return y.astype(np.int)
    # return y.astype(np.int32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


# 3.4.2
def identity_function(x):
    return x

# 3.4.3
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])
    return network


# 3.4.3
def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


# 3.5.1/3.5.2
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    #
    # x = np.array([-1.0, 1.0, 2.0])
    # x = np.arange(-5.0, 5.0, 0.1)
    '''三选一'''
    # y = step_function(x)
    # y = sigmoid(x)
    # y = relu(x)
    '''三选一'''
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)
    # plt.show()
    #
    # x = np.array([-1.0, 1.0, 2.0])
    # sigmoid(x)
    #
    # t = np.array([-1.0, 1.0, 2.0])

    # 3.3.2.matrix multiply
    # A = np.array([[1, 2], [3, 4]])
    # B = np.array([[5, 6], [7, 8]])
    # C = np.array([[1,2,3], [4,5,6]])
    # D = np.array([[1,2], [3,4], [5,6]])
    # np.dot(C, D)

    # 3.3.3神经网络的内积
    X = np.array([1, 2])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    Y = np.dot(X, W)

    # 3.4.三层神经网络的实现
    # X = np.array([1.0, 0.5])
    # W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # B1 = np.array([0.1, 0.2, 0.3])
    # A1 = np.dot(X, W1) + B1
    # Z1 = sigmoid(A1)
    #
    # W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    # B2 = np.array([0.1, 0.2])
    # A2 = np.dot(Z1, W2) + B2
    # Z2 = sigmoid(A2)
    #
    # W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    # B3 = np.array([0.1, 0.2])
    # A3 = np.dot(Z2, W3) + B3
    # Y = identity_function(A3)

    # 3.4.3 代码小结
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

    # 3.5.1
    a = np.array([0.3, 2.9, 4.0])
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    # 3.5.2
    a = np.array([1010, 1000, 990])
    np.exp(a) / np.sum(np.exp(a))
    c = np.max(a)
    a - c
    ans = np.exp(a-c) / np.sum(np.exp(a-c))

    # 3.5.3
    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)
    np.sum(y)



