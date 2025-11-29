import numpy as np

import os,sys
#abolute path 绝对路径,写出__file__文件的完整的路径 __file__是内置变量,表示当前脚本文件目录
    #前后带有'__'表示python自带的魔术方法或内置属性
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.functions import *
from common.util import im2col, col2im

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1,5,5,stride = 1,pad = 0)
print(col1.shape)


#卷积层
class Convolution:
    def __init__(self,W,b,stride = 1,pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        #中间数据(反向传播用)
        self.x = None
        self.col = None
        self.col_W = None
        
        #权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH)/self.stride)
        out_w = int(1 + (W + 2*self.pad - FW)/self.stride)

        col = im2col(x,FH,FW,self.strid,self.pad)
        col_W = self.W.reshape(FN,-1).T#滤波器展开,转置是为了滤波器左乘数据
        out = np.dot(col,col_W) + self.b

        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self,dout):
        FN,C,FH,FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,FN)

        self.db = np.sum(dout,axis = 0)
        self.dW = np.dot(self.col.T,dout)
        self.dW = self.dW.transpose(1,0).reshape(FN,C,FH,FW)

        dcol = np.dot(dout,self.col_W.T)
        dx = col2im(dcol,self.x.shape,FH,FW,self.stride,self.pad)

        return dx
    
#池化层
class Pooling:
    def __init__(self,pool_h,pool_w,stride = 1,pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1+(H - self.pool_h) / self.stride)
        out_w = int(1+(W - self.pool_w) / self.stride)

        #展开
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        #针对输出形状做变换使其从适配滤波器的形状变换为适配适配MAX池化
        col = col.reshape(-1,self.pool_h*self.pool_w)

        #最大值所在索引
        arg_max = np.argmax(col,axis = 1)
        #最大值
        out = np.max(col,axis = 1)
        #转换
        #因为求出来的连起来的一个个元素最里面是深度,所以先排列深度C.后面在重新调整轴方向
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size,pool_size))
        #这里每一行是单通道的池化窗口
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()] = dout.flatten()
        #这里是元组拼接,表示在shape基础上添加一个维度
        #重新配置成(N, out_h, out_w, C, pool_size)
        dmax = dmax.shape(dout.shape + (pool_size,))

        #再配置成col2im能够接受的形状
        #不直接配置是为了减少for循环,这样更快
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2] - 1)
        dx = col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.stride,self.pad)

        return dx