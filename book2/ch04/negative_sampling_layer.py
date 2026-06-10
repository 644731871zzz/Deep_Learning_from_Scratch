import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *
from common.layers import Embedding,SigmoidWithLoss
import collections

class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W) #这里的W没用理论矩阵乘法的形状,和Win的形状一样
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None #存正向传播的数据

    def forward(self,h,idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h,axis = 1) #numpy是对应元素相乘,再相加后会降维
        self.cache = (h,target_W)
        return out
    
    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh