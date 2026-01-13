import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *
from common.config import GPU
from common.functions import softmax,cross_entropy_error


class Matmul:
    def __init__(self,W):
        self.params = [W]#保存参数,这个参数是需要学习的参数
        #因为后面使用了[...],修改的信息重新定位到了开始创建时候的位置
            #这样仅需要在最开始创建的时候确认位置,省去了位置乱的时候grads更新梯度再重新排序的步骤
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self,x):
        W = self.params
        out = np.dot(x,W)
        self.x = x
        return out
    
    def backward(self,dout):
        W = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        #创建时候用的[W],所以0取到了实际的W的位置
        #[...]是深复制,内存位置不改变,内存信息改变
            #不用...将是浅复制,仅仅改变指引用置可能多变量同时指向一个值
            #深复制不改变引用位置,直接在内存位置改值
            #[...]用于索引场景,表示选中这个数据本身的全部数据(包括了内存位置)
                #修改了共享对象的底层数据
        #起到了固定内存地址的作用
            #grads的处理会变简单
        self.grads[0][...] = dW
        return dx
    

class Sigmoid:
    def __init__(self):
        self.params,self.grads = [],[]
        self.out = None
    
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self,dout):
        dx = dout * (1.0 - self.out)*self.out
        return dx
    

class Affine:
    def __init__(self,W,b):
        self.params = [W,b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
    
    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W) + b
        self.x = x
        return out
    
    def backward(self,dout):
        W,b = self.params
        dx = np.dot(dout,W.T)
        dW = np.dot(self.x.T,dout)
        db = np.sum(dout,axis = 0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
    


class SoftmaxWithLoss:
    def __init__(self):
        self.params,self.grads = [],[]
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:#如果相等证明是ont_hot,类别索引,形状一致
            self.t = self.t.argmax(axis = 1)#将其修改为1D,直接得到索引标签的形式

        loss = cross_entropy_error(self.y,self.t)
        return loss

    def backward(self,dout = 1):
        batch_size = self.t.shape[0]#得到样本数

        dx = self.y.copy()#先初始化为输出的y
        #在正确的项上-1,因为正确的t就位1,导数直接减1,错误的为0,不需要减
        #高级索引,逐行对应元素,传入了两个数列,一一对应行列找到元素进行处理(每一行的一个列)
            #如果使用[:,[]],将对所有行下的[]中的所有列索引进行操作(所有行某些列)
        dx[np.arange(batch_size),self.t] -= 1#到这仅仅是当前的局部导数,还没乘以反向上游传递下来的
        #
        dx *= dout 
        #
        dx = dx / batch_size

        return dx