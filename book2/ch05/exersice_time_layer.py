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

    def forward(self,xs):#多的s是sequence的意思:一系列
        Wx,Wh,b = self.params
        #有批次,Time,单词自身
        N,T,D = xs.shape
        D,H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H),dtype = 'f') #为输出准备容器,初始化了一些没有意义的值存在了里面
        
        #首次调用时候初始化h
        if not self.stateful or self.h is None:
            self.h = np.zeros((N,H),dtyep = 'f')

        for t in range(T):
            layer = RNN(*self.params)
            #这里的块大小按照反向传播的块的大小定的
            #下一次调用forward时候,self.h没有被删,如果stateful为True,那么继续传递
            #注意这里是传入多少xs创建多少的layer,不是固定layer
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)

        return hs
    
    def backward(self,dhs):
        Wx,Wh,b = self.params
        N,T,H = dhs.shape
        D,H = Wx.shape

        dxs = np.empty((N,T,D),dtype = 'f')
        dh = 0
        grads = [0,0,0] 
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx,dh = layer.backward(dhs[:,t,:] + dh)
            dxs[:,t,:] = dx

            for i ,grad in enumerate(layer.grads):
                grads[i] += grad

        for i,grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self,h):
        self.h = h
    
    def reset_state(self):
        self.h = None


class TimeAffine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None

    def forward(self,x):
        N,T,D = x.shape
        W,b = self.params

        rx = x.reshape(N*T,-1) #先走完N = 0的然后拼接走N = 1的T,所以这里按一簇一簇的拼接
        out = np.dot(rx,W) + b
        self.x = x
        return out.reshape(N,T,-1)
    
    def backward(self,dout):
        x = self.x
        N,T,D = x.shape
        W,b = self.params

        dout = dout.reshape(N*T,-1)
        rx = x.reshae(N*T,-1)

        db = np.sum(dout,axis = 0)
        dW = np.dot(rx.T,dout)
        dx = np.dot(dout,W.T)
        #所以这里*要破坏元组的外壳,只给里面的值.类似(1,2,3)>>>1,2,3,恰好去对应位置参数
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx
    

class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [],[]
        self.cache = None
        self.ignore_label = -1 #忽略标签

    def forward(self,xs,ts):
        N,T,V = xs.shape

        #监督数据为one-hot的情况下
        #这里无法帮助ts补齐(补-1)
        if ts.ndim == 3:
            ts = ts.argmax(axis = 2)

        mask = (ts != self.ignore_label) #判断那些不是-1 #-1是补齐用矩阵的没有用

        xs = xs.reshape(N*T,V)
        ts = ts.reshape(N*T)
        mask = mask.reshape(N*T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N*T),ts]) #ys这里取出正确项对应的概率,这是个索引语法
        ls *= mask #忽略ts为-1的地方,利用布尔数False为0的特性清除
        loss = -np.sum(ls) #符号是损失函数的公式带的,这里补上
        loss /= mask.sum() #用布尔数组为1的特性,求和后取平均

        self.cache = (ts,ys,mask,(N,T,V))
        return loss
    
    def backward(self,dout = 1):
        ts,ys,mask,(N,T,V) = self.cache

        dx = ys
        dx[np.arange(N*T),ts] -= 1 #正确标签的位置 - 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:,np.newaxis] #将无效位置清零

        dx = dx.reshape((N,T,V))

        return dx