import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *
from common.layers import *
from common.functions import softmax,sigmoid

class RNN:
    def __init__(self,Wx,Wh,b):
        self.params = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        self.cache = None #

    def forward(self,x,h_prev):#
        Wx,Wh,b = self.params
        t = np.dot(h_prev,Wh) + np.dot(x,Wx) + b
        h_next = np.tanh(t)

        self.cache = (x,h_prev,h_next) #
        return h_next