import os,sys
#abolute path 绝对路径,写出__file__文件的完整的路径 __file__是内置变量,表示当前脚本文件目录
    #前后带有'__'表示python自带的魔术方法或内置属性
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class SimpleConvNet:
    def __init__(self,input_dim = (1,28,28),
                 conv_param = {'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size = 100,output_size = 10,weight_init_std = 0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]#高宽相同,取一个
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1#高宽相同求一次
        pool_output_size = int(filter_num * (conv_output_size/2)*(conv_output_size/2))#简单的2*2池化,高宽变为1/2,池化层深度和滤波器数量一致

        #初始化设置
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num,input_dim[0],filter_size,filter_size)#最基础的标准正太分布,外部乘以和加减自定义,里面的参数仅为配置矩阵形状
        self.params['b1'] = np.zeros(filter_num)#每个偏置对应了一个滤波器的输出,不是所有滤波器都是相同的
        self.params['W2'] = weight_init_std*\
                            np.random.randn(pool_output_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std*\
                            np.random.randn(hidden_size,output_size)
        self.params['b3'] = np.zeros(output_size)

        #生成层
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],
                                           conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = Relu()#不需要权重和形状,向前时通过传进来的值直接计算
        self.layers['Pool1'] = Pooling(pool_h = 2,pool_w = 2,stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])

        self.last_layer = SoftmaxWithLoss()#不需要权重和形状,通过传进来的值


    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    #为了反向传播,模型可用
    def loss(self,x,t):
        y = self.predict(x)
        return self.last_layer.forward(y,t)
    
    #计算成功率,为了人看的方便,loss的设计是为了反向传播可导
    def accuracy(self,x,t,batch_size = 100):
        if t.ndim != 1 : t = np.argmax(t,axis = 1)#如果不满足if自动继续执行
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]#索引切片只写一层表示沿着第0维度取值,如果给定范围超出将会自动截停导末尾
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y,axis = 1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
    

    def numerical_gradient(self,x,t):
        #w是形式参数,没有被用到,都会执行返回值(求loss)
        loss_w = lambda w:self.loss(x,t)#

        grads = {}
        for idx in (1,2,3):
            #传入的第二个参数被修改了内存,导致每次推导不一样用于求导
            grads['W' + str(idx)] = numerical_gradient(loss_w,self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w,self.params['b' + str(idx)])

        return grads
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout = 1
        dout = self.last_layer.backward(dout)

        #内层self.layers.values()返回可迭代对象,list将返回的对象变为列表
        #可迭代对象可以直接在list()中自动迭代运行收集
            #但是不用用[]直接括起来,这样迭代对象本身会变为元素
        layers = list(self.layers.values())
        layers.reverse()#将列表翻转用于反向传播

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'],grads['b1'] = self.layers['Conv1'].dW,self.layers['Conv1'].db
        grads['W2'],grads['b2'] = self.layers['Affine1'].dW,self.layers['Affine1'].db
        grads['W3'],grads['b3'] = self.layers['Affine2'].dW,self.layers['Affine2'].db

        return grads


    def save_params(self,file_name = 'params.pkl'):
        params = {}
        for key,val in self.params.items():
            params[key] = val
        #创建环境,先打开文件对象,并将对象绑定到变量名,类似'='
        #'wb'是打开模式,w表示写入,b表示二进制
        #如果文件存在,清空覆盖,如果不存在创建
        with open(file_name,'wb') as f:
            #写入文件,f是文件对象
            #pickle负责将python对象转换为二进制流,在文件可写入的地方写入二进制流
            pickle.dump(params,f)

    def load_params(self,file_name = 'params.pkl'):
        with open(file_name,'rb') as f:
            #读取二进制流并转换为python对象
            params = pickle.load(f)
        for key,val in params.items():
            self.params[key] = val

        for i,key in enumerate(['Conv1','Affine1','Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
