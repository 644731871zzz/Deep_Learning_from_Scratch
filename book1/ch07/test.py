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