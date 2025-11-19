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

        #初始化设置
        self.__init_weight(weight_init_std)#

        #生成层
        activation_layer = {'sigmoid':Sigmoid,'relu':Relu}
        self.layers = OrderedDict()
        for idx in range(1,self.hidden_layer_num+1):
            