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
    return x_test,t_test

#读取保存在命名的pickle文件中的权重参数(这里假设学习已经完成)
def init_network():
    #进入with时候,执行with后函数中的enter,会返回一个值给as后面的变量
        #然后在with as下的函数使用as后的变量进行执行
        #执行with后的函数的exit函数清理with程序(释放资源),不管as后面的变量(变量值不变)
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

#t表示target (目标标签)
x,t = get_data()
network = init_network()
accuracy_cnt = 0 #初始化测试概率值
for i in range(len(x)):
    y = predict(network,x[i]) #numpy数组形式输出各个标签的概率,索引位置与数字一一对应
    p = np.argmax(y) #获取概率最高的元素的索引 argument
    if p == t[i]:
        accuracy_cnt += 1

print('Accuracy:' + str(float(accuracy_cnt) / len(x)))