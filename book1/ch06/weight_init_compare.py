import os,sys
current_script_path = os.path.abspath(__file__)#
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.optimizer import *
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

#将0-255的数据进行了归一化范围0-1
(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


weight_init_types = {'std=0.01':0.01,'Xavier':'sigmoid','He':'relu'}
optimizer = SGD(lr = 0.01)

networks = {}
train_loss = {}
for key,weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size = 784,hidden_size_list = [100,100,100,100],
                                  #最后一个参数可以被传入字符串,这个类被特殊设计
                                    #如果是数字,自动按照标准差生成权重
                                    #如果是字符串,将按照激活函数字符串所对应初始化模式的生成权重
                                        #这里仅仅是配置生成初始权重的方式,没有配置激活函数是什么
                                        #所以可以用ReLU激活函数并用sigmoid对应的Xavier模式生成初始权重
                                  output_size = 10,weight_init_std = weight_type)
    train_loss[key] = []


for i in range(max_iterations):
    #又放回抽样,从0到train_size-1的范围,左闭右闭
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch,t_batch)
        optimizer.update(networks[key].params,grads)

        #从第一次更新参数后开始记录,没有记录参数更新前的初始权重的loss
        loss = networks[key].loss(x_batch,t_batch)
        train_loss[key].append(loss)

    if i % 100 ==0:
        print('===========' + 'iteration:' + str(i) + '===========')
        for key in weight_init_types.keys():
            loss = networks[key].loss(x_batch,t_batch)
            print(key + ':' + str(loss))


markers = {'std=0.01':'o','Xavier':'s','He':'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    #最后一个参数是图例标签
    plt.plot(x,smooth_curve(train_loss[key]),marker = markers[key],markevery = 100,label = key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0,2.5)
plt.legend()
plt.show()


