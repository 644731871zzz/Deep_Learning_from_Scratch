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
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD,Adam

(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True)

#减少学习数据
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

#两个下划线为了表示这是内部使用的函数,不是左右各两个的python魔术名
def __train(weight_init_std):
    bn_network = MultiLayerNetExtend(input_size = 784,hidden_size_list = [100,100,100,100,100],output_size = 10,
                                     weight_init_std = weight_init_std,use_batchnorm = True)#添加了新的层
    network = MultiLayerNetExtend(input_size = 784,hidden_size_list=[100,100,100,100,100],output_size = 10,
                                  weight_init_std = weight_init_std)
    optimizer = SGD(lr = learning_rate)

    train_acc_list = []#记录准确率
    bn_train_acc_list = []

    #每个纪元多少次迭代
    iter_per_epoch = max(train_size / batch_size,1)
    epoch_cnt = 0 #纪元计数

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network,network):#
            grads = _network.gradient(x_batch,t_batch)
            optimizer.update(_network.params,grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train,t_train)
            bn_train_acc = bn_network.accuracy(x_train,t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print('epoch:' + str(epoch_cnt) + '|' + str(train_acc) + '-' + str(bn_train_acc))

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list,bn_train_acc_list


#是log不是lin 所以生成的范围是10^0 -- 10^-4 的16个数,按照对数间隔均匀分布(指数位置)
weight_scale_list = np.logspace(0,-4,num = 16)
x = np.arange(max_epochs)

#返回index和value index从0开始
for i,w in enumerate(weight_scale_list):
    print('===========' + str(i+1) + '/16' + '==========')
    train_acc_list,bn_train_acc_list = __train(w)#

    #subplot索引从开始是从1开始,所以+1
    #先使用布局,然后激活索引位置的子图
        #如果没有此布局fig,先创建,如果有了此布局相同的布局,复用布局,然后用索引定位
    plt.subplot(4,4,i+1)
    #不同标准差下训练的数据的标题
    plt.title('W:' + str(w))
    #i = 15是最后一样图,使用label表示在最后一张图上添加图里,防止图例过多
    if i == 15:
        plt.plot(x,bn_train_acc_list,label = 'Batch Normalization',markevery = 2)#
        plt.plot(x,train_acc_list,linestyle = '--',label = 'Normal(without BatchNorm)',markevery = 2)
    else:
        plt.plot(x,bn_train_acc_list,markevery = 2)
        plt.plot(x,train_acc_list,linestyle = '--',markevery = 2)

    plt.ylim(0,1.0)
    #如果不为0将不是最左边的列,执行if下的语句将刻度隐藏
    if i % 4:
        plt.yticks([])
    #如果上面判断为0,表示是最左列,取消刻度隐藏刻度,配置标签
    else:
        plt.ylabel('accuracy')
    #当i<12时候表示不是最下面第四行,不显示x刻度
    if i<12:
        plt.xticks([])
    #当i≥12时候是最下面四行,取消不显示刻度的设置,并配置标签
    else:
        plt.xlabel('epochs')
    plt.legend(loc = 'lower right')

plt.show()