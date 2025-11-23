import os,sys
#abolute path 绝对路径,写出__file__文件的完整的路径 __file__是内置变量,表示当前脚本文件目录
    #前后带有'__'表示python自带的魔术方法或内置属性
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True)#

#实现高速化,减少训练数据
x_train = x_train[:500]
t_train = t_train[:500]

#分割验证数据
validation_rate = 0.20
#多少个
validation_num = int(x_train.shape[0] * validation_rate)
x_train,t_train = shuffle_dataset(x_train,t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr,weight_decay,epocs = 50):
    network = MultiLayerNet(input_size=784,hidden_size_list=[100,100,100,100,100,100],
                            output_size=10,weight_decay_lambda=weight_decay)
    trainer = Trainer(network,x_train,t_train,x_val,t_val,
                      epochs = epocs,minibatch_size = 100,
                      optimizer = 'sgd',optimizer_param={'lr':lr},verbose = False)
    trainer.train()

    return trainer.test_acc_list,trainer.train_acc_list

#超参数搜索
optimization_trial = 100#
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    weight_decay = 