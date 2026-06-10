from common.np import *
from common.config import GPU
from common.functions import softmax,cross_entropy_error

class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self,idx):
        W = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    # def backward(self,dout):
    #     dW = self.grads
    #     dW[...] = 0 #对所有维度的值赋予0 dW[]表示numpy的赋值规则
    #     dW[self.idx] = dout
    #     return None

    def backward(self,dout):
        dW = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW,self.idx,dout)
        else:
            #at会在W在self.idx的对应位置加上dout
            #这里注意self.idx和dout数量是对应的
            #注意如果出现相同的位置,对应位置将会累加
            #这里比for循环处理速度更快
            np.add.at(dW,self.idx,dout)
        return None