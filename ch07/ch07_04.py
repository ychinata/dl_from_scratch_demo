# coding: utf-8
# 2022.6.3
# xy
# 卷积层和池化层的实现
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from common_all.util import im2col


# 7.4.1 四维数组
x = np.random.rand(10, 1, 28, 28)
print(x.shape)
# 访问第n个数据
print(x[0].shape)
print(x[1].shape)
# 访问第1个数据第1个通道的空间数据
print(x[0, 0].shape)

# 7.4.3 卷积层的实现
x1 = np.random.rand(1, 3, 7, 7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)  # (9, 75)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)  # (90, 75)

