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
            per_channel_num = conv_param['filter_num']#赋值下一层需要用到的通道数量
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4,hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size,output_size)
        self.params['b8'] = np.zeros(output_size)


        #生成层
        self.layers = []
        self.layers.append(Convolution(self.params['W1'],self.params['b1'],
                                       conv_param_1['stride']),conv_param_1['pad'])
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'],self.params['b2'],
                                       conv_param_2['stride']),conv_param_2['pad'])
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Convolution(self.params['W3'],self.params['b3'],
                                       conv_param_3['stride']),conv_param_3['pad'])
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'],self.params['b4'],
                                       conv_param_4['stride']),conv_param_4['pad'])
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Convolution(self.params['W5'],self.params['b5'],
                                       conv_param_5['stride']),conv_param_5['pad'])
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'],self.params['b6'],
                                       conv_param_6['stride']),conv_param_6['pad'])
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2,pool_w = 2,stride = 2))

        self.layers.append(Affine(self.params['W7'],self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'],self.params['b8']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()


        
            