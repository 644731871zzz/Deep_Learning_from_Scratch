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
                 hidden_size = 100,output_size = 10,weight_init_std = 1):
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
        self.layers = OrderedDict
        self.layers['Conv1'] = Convolution(self.params['W1'],self.params['b1'],
                                           conv_param['stride'],conv_param['pad'])
        self.layers['Relu1'] = Relu()#不需要权重和形状,向前时通过传进来的值直接计算
        self.layers['Pool1'] = Pooling(pool_h = 2,pool_2 = 2,stride = 2)
        self.layers['Affine1'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],self.params['b3'])

        self.last_layer = SoftmaxWithLoss()#不需要权重和形状,通过传进来的值


