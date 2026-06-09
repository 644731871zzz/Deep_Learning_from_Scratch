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
    
    def backward(self,dout):
        dW = self.grads
        dW[...] = 0 #
        dW[self.idx] = dout
        return None