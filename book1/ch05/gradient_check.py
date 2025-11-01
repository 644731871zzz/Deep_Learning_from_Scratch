import sys, os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在目录
current_dir = os.path.dirname(current_script_path)

# 将 "book1" 目录（您的项目根目录）添加到 sys.path
# 假设您的文件结构是 .../book1/ch05/gradient_check.py
# 那么父目录的父目录就是 book1
parent_dir = os.path.dirname(current_dir)

# 将 book1 目录添加到 sys.path
sys.path.append(parent_dir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

#读取数据
(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True,one_hot_label=True)

network = TwoLayerNet(input_size = 784,hidden_size = 50,output_size = 10) #

x_batch = x_train[:3]#
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch,t_batch)
grad_backprop = network.gradient(x_batch,t_batch) #

for key in grad_numerical.keys(): #
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff)) #