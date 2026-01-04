import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    def __init__(self,input_dim = (1,28,28),
                 conv_param_1 = {'filter_num':16,'filter_size':3,'pad':1,'stride':1},
                 conv_param_2 = {'filter_num':16,'filter_size':3,'pad':1,'stride':1},
                 conv_param_3 = {'filter_num':32,'filter_size':3,'pad':1,'stride':1},
                 conv_param_4 = {'filter_num':32,'filter_size':3,'pad':2,'stride':1},
                 conv_param_5 = {'filter_num':64,'filter_size':3,'pad':1,'stride':1},
                 conv_param_6 = {'filter_num':64,'filter_size':3,'pad':1,'stride':1},
                 hidden_size = 50,output_size = 10):
        
        #初始化权重
        #各层神经元平均与前一层的几个神经元有连接
        pre_node_nums = np.array([1*3*3,16*3*3,16*3*3,32*3*3,32*3*3,64*3*3,64*4*4,hidden_size])
        wight_init_scales = np.sqrt(2.0/pre_node_nums)#使用Relu下用He初始值

        self.params = {}
        pre_channel_num = input_dim[0]#初始化用输入数据的通道数量
        for idx,conv_param in enumerate([conv_param_1,conv_param_2,conv_param_3,conv_param_4,conv_param_5,conv_param_6]):
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],pre_channel_num,
                                                                                     conv_param['filter_size'],conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']#赋值下一层需要用到的通道数量
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4,hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size,output_size)
        self.params['b8'] = np.zeros(output_size)


        #生成层
        self.layers = []
        self.layers.append(Convolution(self.params['W1'],self.params['b1'],
                                       conv_param_1['stride'],conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'],self.params['b2'],
                                       conv_param_2['stride'],conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Convolution(self.params['W3'],self.params['b3'],
                                       conv_param_3['stride'],conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'],self.params['b4'],
                                       conv_param_4['stride'],conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Convolution(self.params['W5'],self.params['b5'],
                                       conv_param_5['stride'],conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'],self.params['b6'],
                                       conv_param_6['stride'],conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Affine(self.params['W7'],self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'],self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self,x,train_flg = False):#train_flg是训练状态参数 因为是预测过程,不需要开启
        for layer in self.layers:
            #Dropout这个层必须知道是不是在训练过程,train_flg是必须的.创建类的时候没有规定默认参数就是必须传入的参数
            if isinstance(layer,Dropout):#isinstance是判断类型函数,判断layer是不是Dropout类的实例
                x = layer.forward(x,train_flg)#如果layer是dropout类实例,参数train_flg为true会丢弃神经元
            else:
                x = layer.forward(x)

        return x
    
    def loss(self,x,t):
        y = self.predict(x,train_flg = True)
        return self.last_layer.forward(y,t)
    
    def accuracy(self,x,t,batch_size = 100):
        if t.ndim != 1 : t = np.argmax(t,axis = 1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):#舍弃了凑不够100个的簇,int舍弃'.'后的值 (向下取整,取更小的数)
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx,train_flg = False)
            y = np.argmax(y,axis = 1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
    
    def gradient(self,x,t):
        #forward一次,存储一次值
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()#反转顺序

        for layer in tmp_layers:
            dout = layer.backward(dout)#dw db自动保存,这里是将dout输出传给反向传播的下一层(正向的上一层)

        grads = {}
        #只取出来需要进行梯度更新的层,卷积层和全连接层,(不取激活层,池化层,dropout层)
        for i,layer_idx in enumerate((0,2,5,7,10,12,15,18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads
    

    def save_params(self,file_name = 'params.pkl'):
        params = {}
        for key,val in self.params.items():
            params[key] = val
        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name = 'params.pkl'):
        with open(file_name,'rb') as f:
            params = pickle.load(f)
        for key ,val in params.items():
            self.params[key] = val

        for i,layer_idx in enumerate((0,2,5,7,10,12,15,18)):
            #将初始化的参数写入到对应层中
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]