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
        self.cache = None #缓存,存放反向传播的数据

    def forward(self,x,h_prev):
        Wx,Wh,b = self.params
        t = np.dot(h_prev,Wh) + np.dot(x,Wx) + b
        h_next = np.tanh(t)

        self.cache = (x,h_prev,h_next) 
        return h_next
    
    def backward(self,dh_next):
        Wx,Wh,b = self.params
        x,h_prev,h_next = self.cache

        dt = dh_next * (1 - h_next **2)
        db = np.sum(dt,axis = 0)
        dWh = np.dot(h_prev.T,dt)
        dh_prev = np.dot(dt,Wh.T)
        dWx = np.dot(x.T,dt)
        dx = np.dot(dt,Wx.T)

        self.grads[0][...] = dWx #替换旧数据
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx,dh_prev
    

class TimeRNN:
    #stateful表示当前状态,如果为True,每个分块都传播上一个h,如果为False,每个分块第一个Rnn不接收h
        #隐藏状态可以理解为上一步留下来的记忆
    def __init__(self,Wx,Wh,b,stateful = False):
        self.patams = [Wx,Wh,b]
        self.grads = [np.zeros_like(Wx),np.zeros_like(Wh),np.zeros_like(b)]
        #保存多个RNN层
        self.layers = None

        #self.h保存本块最后一个h self.dh保存传给前一块的梯度
        self.h,self.dh = None,None
        self.stateful = stateful

    def forward(self,xs):#
        Wx,Wh,b = self.params
        N,T,D = xs.shape
        D,H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H),dtype = 'f') #
        
        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H),dtyep = 'f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs




    def set_state(self,h):
        self.h = h
    
    def reset_state(self):
        self.h = None