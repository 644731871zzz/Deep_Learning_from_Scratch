import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


(x_train,t_train),(x_test,t_test) = load_mnist(flatten = False)

network = DeepConvNet()
network.load_params('ch08/deep_convnet_params.pkl')

sampled = 10000
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print('caluculate accuracy (float64) ... ')
print(network.accuracy(x_test,t_test))

#转换为16位
x_test = x_test.astype(np.float16)
for params in network.params.values():
    params[...] = params.astype(np.float16)

print('caluculate accuracy (float16) ... ')
print(network.accuracy(x_test,t_test))