import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:
    #初始化,参数除了self依次表示为输入层神经元数,隐藏层神经元数,输出层神经元数,权重初始化标准差
    #初始化就是生成时被调用的方法
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        #初始化权重
        self.params = {}
        #randn生成正太分布随机数,如果想指定均值和标准差,需在外侧进行线性变换(像是rand)
        #默认均值为0,标准差为1
        #这里的调整标准差
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    #计算神经网络推理
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y
    
    #计算损失函数 x为输入数据t为监督数据(正确解标签)
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t) #交叉熵误差法
    
    #计算识别精度
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1) #返回最大索引位置,axis = 1表示在每一行中搜寻返回一个数组
        t = np.argmax(t,axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0]) #y==t是numpy数组的比较广播(前提是y和t形状相同)
        return accuracy
    
    #计算权重参数梯度
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        
        #保存梯度的字典型变量
        grads = {}
        #第一层权重的梯度
        #这里调用的函数不是这里def定义的函数自循环
            #调用的是之前from那里导入的函数
            #如果自循环是self.numerical_gradient才是(类需要self才能调用自身)
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])#基于数值微分法计算梯度
        #第一层偏置的梯度
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads

    #计算权重参数梯度(高速版)(节约时间)
    def gradient(self,x,t):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        a1 = np.dot(x,W1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2)+b2
        y = softmax(a2)

        dy = (y - t) / batch_num #
        grads['W2'] = np.dot(z1.T,dy)
        grads['b2'] = np.sum(dy,axis = 0)

        da1 = np.dot(dy,W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T,dz1)
        grads['b1'] = np.sum(dz1,axis = 0)

        return grads

