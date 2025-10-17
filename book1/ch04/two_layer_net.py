import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    #计算神经网络输出
    def predict(self,x):
        W1,W2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y
    
    #计算损失函数 x为输入数据t为监督数据
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t) #
    
    #
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1)
        t = np.argmax(t,axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy



