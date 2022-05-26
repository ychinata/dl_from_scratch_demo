# coding: utf-8
# 2022.5.28
# xy
# Affine/Softmax层的实现

import numpy as np

# 5.6.2批版本的Affine层
# 正向传播
X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])
B = np.array([1, 2, 3])
print(X_dot_W)
print(X_dot_W + B)

# 反向传播
dY = np.array([[1, 2, 3], [4, 5, 6]])
print(dY)
dB = np.sum(dY, axis=0)  # 按行维度求和
print(dB)
