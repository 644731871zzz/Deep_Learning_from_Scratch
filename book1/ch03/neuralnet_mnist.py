import sys,os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid,softmax

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(normalize = True ,
                                                   flatten = True ,
                                                   one_hot_label = False)
    retrun x_test,t_test

#读取保存在命名的pickle文件中的权重参数(这里假设学习已经完成)
def init_network():
    #with,as是向下问管理器,自动负责打开和关闭 
    ##with后的对象将会命名为as后的变量,用完这个变量自动清理对象
    #open为python内置函数,用来打开文件open('文件名','模式')  rb表示二进制
    with open('sample_weight.pkl','rb') as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y