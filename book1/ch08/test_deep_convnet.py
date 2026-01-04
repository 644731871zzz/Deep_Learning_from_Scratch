import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
print(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer


(x_train,t_train),(x_test,t_test) = load_mnist(flatten = False)

network = DeepConvNet()
network.load_params('ch08/deep_convnet_params.pkl')#python读取文件自带根据字符串寻找文件夹最终定位到文件的功能
accuracy = network.accuracy(x_test,t_test)
print(accuracy)