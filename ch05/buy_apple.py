# coding: utf-8
from layer_naive import *
# 2022.4.10
# hxy review

apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
# print(price)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice) # 1.1 * 200
dapple, dapple_num = mul_apple_layer.backward(dapple_price) # 2 * 100

print("price:", int(price))
print("dapple_price:", dapple_price)
print("dTax:", dtax)
print("dApple:", dapple)
print("dApple_num:", int(dapple_num))

