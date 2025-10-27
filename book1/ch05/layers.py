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
    

class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forwarid(self,x):
        self.x = x
        out = np.dot(x + self.W) + self.b

        return out
    
    def backward(self,dout):
        dx = np.dot(dout,self.W.T)
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis = 0)

        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #损失
        self.y = None #softmax输出
        self.t = None #监督数据(one - hot vector)

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)

        return self.loss
    
    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        #因为是一簇数据,一簇训练数据的损失函数的输出还要进行相加平均
        #反向传播就把求平均传递回来了
        dx = (self.y - self.t) / batch_size

        return dx