import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid,softmax

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize = True,
                                                   flatten = True,
                                                   one_hot_label = False)
    return x_test,t_test

def init_network():
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    w1,w2,w3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,w3) + b3
    y = softmax(a3)

    return y

x,t = get_data()
network = init_network()

batch_size = 100 #批数量
accuracy_cnt = 0 #初始化正确数量

for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size] #索引提取是左闭右开
    y_batch = predict(network,x_batch)
    #判断每一行,axis为二维数组的最内侧,列方向,每一行
    p = np.argmax(y_batch,axis = 1)
    #使用 == 判断p数组和索引出对应位置的数组对比生成布尔数组,然后sum求和True(当成1)后累加
        #==本身是python中的比较运算符,作用在numpy数组会重载运算符实现逐元素比较返回布尔数组
        #只要有一方是numpy数组直接重载运算符.python看到==时候会优先调用对象的特殊方法
        #==本身是语法糖,==是调用对象的一个方法当有一方是numpy将会使用numpy数组的__eq__方法
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print('Accuracy:' + str(float(accuracy_cnt)/len(x)))