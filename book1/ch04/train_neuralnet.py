import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

#train是训练的意思 来自training(训练的意思)
#第一个参数表示归一化,0-255变为0-1
#第二个参数表示将标签转位独特向量 如2>>>[0,0,1,0,0,0] 就是对应索引位置标记为1
(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True,one_hot_label = True)

network = TwoLayerNet(input_size = 784,hidden_size = 50,output_size = 10)

#超参数
iters_num = 10000 #迭代器次数
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1 #学习的更新幅度

print(train_size)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#平均每个epoch的重复次数
#max返回两者值中的较大值,保证结果至少为1
iter_per_epoch = max(train_size / batch_size,1)


for i in range(iters_num):
    #获取mini_batch
    #从总的训练数据中随机抽取一部分训练数据
    #choice的参数中表示从[0,1,2...,train_size - 1]中随机挑batch_size个索引
        #放回抽样,可重复,范围是左闭右开 [0,train_size) 个数是train_size,但是从0开始到train_size - 1
        #返回整数数组
    batch_mask = np.random.choice(train_size,batch_size)
    #mask是掩码的意思,表示筛选或者遮盖部分数据的索引
    #本意是遮盖数据的意思,能够筛选是因为遮盖住了其他数据筛出需要的数据
        #遮盖仅仅是抽象背景了,编程中简化成筛出想要的部分
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #计算梯度
    #grad = network.numerical_gradient(x_batch,t_batch)
    grad = network.gradient(x_batch,t_batch)#高速版

    for key in ('W1','b1','W2','b2'):
        network.params[key] -= learning_rate * grad[key]

    #记录学习过程
    #求出损失函数
    loss = network.loss(x_batch,t_batch)
    #记录每一次的损失函数
    train_loss_list.append(loss)

    #计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        #str将其变成字符串后然后拼接输出
        print('train acc, test acc | ' + str(train_acc) + ',' + str(test_acc))


#绘制图像
markers = {'train':'o','test':'s'}
x = np.arange(len(train_acc_list))
plt.plot(x,train_acc_list,label = 'train acc')
plt.plot(x,test_acc_list,label = 'test acc',linestyle = '--')
plt.xlabel('epochs')
plt.ylabel('accuracy') #识别精度标题
plt.ylim(0,1.0)
#显示图例 loc配置图例位置
plt.legend(loc = 'lower right')
plt.show()