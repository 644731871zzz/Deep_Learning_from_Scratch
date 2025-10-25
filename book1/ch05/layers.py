import numpy as np
import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.util import im2col,col2im


class Relu:
    #概念上类赋予给了一个变量名是一个单独的类
    #但是实际编程中是变量名是实例自动传入的位置参数.
        #apple.forward(2,3) 等价于>> MulLayer.forward(apple,2,3)
    def __init__(self):
        self.mask = None

    def forward(self,x):
        #保存为numpy类型的true/false数组
        #如果小于等于0存为true,形状与x相同
        self.mask = (x<=0)
        out = x.copy()
        #布尔掩码索引,如果out和self.mask形状形同,对true的地方进行操作
        #适用于任意维度的数组 不需要用切片形式
        out[self.mask] = 0

        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx