import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000,100)
node_num = 100 #一层的节点个数
hidden_layer_size = 5
activations = {} #保存激活值结果

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    #w = np.random.randn(node_num,node_num) *1
    w = np.random.randn(node_num,node_num) *0.01
    #w = np.random.randn(node_num,node_num) * np.sqrt(1.0 / node_num)
    #w = np.random.randn(node_num,node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x,w)

    z = sigmoid(a)
    #z = ReLU(a)
    #z = tanh(a)

    #key可以是任何哈西对象(整数,字符串,元组),不会自动转换为字符串
    #能通过哈希找到的固定数值的对象-只要对象内容不可变就可以作为字典
        #同一个不可变对象的哈希永远相同 不随着变量绑定而变化
    activations[i] = z #

for i,a in activations.items(): #返回的key到i key保持原类型
    #没图自动按照子图指定布局创建,只有一行,多少层就有多少列 第三个参数指向子图
    #如果有图了先检查是否有函数参数同布局图,如果有 第三个参数就是指向子图
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1)+'-layer')
    if i!=0: plt.yticks([],[])#第一个[]去掉刻度线 第二个[]去掉刻度标签(文字)
    #将数据展开,直方图分为30个区间,只统计0-1之间的值(闭区间)
    plt.hist(a.flatten(),30,range = (0,1))
plt.show()