import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNet:
    """全连接多层神经网络"""
    def __init__(self,input_size,hidden_size_list,output_size,
                 activation = 'relu',weight_init_std = 'relu',weight_decey_lambda = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decey_lambda
        self.params = {}

        #初始化设置,生成权重
        self.__init_weight(weight_init_std)

        #生成层
        activation_layer = {'sigmoid':Sigmoid,'relu':Relu}
        self.layers = OrderedDict()
        #这里是讲layer做成一个大的层,仿射层中的每一层中间都夹着激活函数层的一层
        for idx in range(1,self.hidden_layer_num+1):
            #仿射层,这是利用权重创建仿射层,而不是生成权重
            #封装成一个对象,正向传播和反向转播都好求
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                      self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1#补齐输出层对应的仿射层,也是加到仿射层中
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)],
                                                  self.params['b'+str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self,weight_init_std):
        """设定权重初始值"""
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        #range(1,n)最后一个是n-1(这里不是range(10),这个输出0-9)
        for idx in range(1,len(all_size_list)):
            scale = weight_init_std
            #.lower()将输入都变成小写字符,是字符串方法  in判断是否为两者之一
            if str(weight_init_std).lower() in ('relu','he'):
                scale = np.sqrt(2.0/all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid','xavier'):
                scale = np.sqrt(1.0/all_size_list[idx - 1])

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1],all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self,x):
        for layer in self.layer.values():
            x = layer.forward(x)

        return x
    
    def loss(self,x,t):
        y = self.predict(x)

        weight_decay = 0#创建正则项
        #隐藏层是N 算上输出层是N+1 在算上range从1到n-1 所以是N+2
        for idx in range(1,self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y,t) + weight_decay

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1)
        #如果标签不是一维,说明是one - hot ,将正确的索引值取出来
        #简写了,没有换行
        if t.ndim != 1 : t = np.argmax(t,axis = 1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        #创建匿名函数,但是W没有用,只是接口需要
        loss_W = lambda W :self.loss(x,t)

        grads = {}
        #从1到N是隐藏层数量 N+1是最后输出层 最后在+1是因为range的限制
        for idx in range(1,self.hidden_layer_num +2):
            grads['W' + str(idx)] = numerical_gradient(loss_W,self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W,self.params['b' + str(idx)])

        return grads
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1#反向传播的起点,只是占的位置
        #输入dout会自动得到输出层的反向传播的导数
        dout = self.last_layer.backward(dout)

        #提取出来的不是映射层或者激活函数层的某个W或者其他
        #是提出出来layers中的建立的类对象,因为一层就是一层,一层是一个类,这个字典指向了这个类
        layers = list(self.layers.values())
        #将数列顺序颠倒
        layers.reverse()
        for layer in layers:
            #每层回传dout进行反向传播的计算
            #值存储在映射层的.dW中了
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1,self.hidden_layer_num + 2):
            #将算出来的dW加上正则
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db
        
        return grads