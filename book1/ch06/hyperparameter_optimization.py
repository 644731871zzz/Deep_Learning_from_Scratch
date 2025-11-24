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
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

#虽然右边返回一个大元组,但是赋值操作中,python会将带','的部分外面也再扩大成一个大元祖结果就是形状匹配
(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True)#

#实现高速化,减少训练数据
x_train = x_train[:500]
t_train = t_train[:500]

#分割验证数据
validation_rate = 0.20
#多少个
validation_num = int(x_train.shape[0] * validation_rate)
x_train,t_train = shuffle_dataset(x_train,t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr,weight_decay,epocs = 50):
    network = MultiLayerNet(input_size=784,hidden_size_list=[100,100,100,100,100,100],
                            output_size=10,weight_decay_lambda=weight_decay)
    trainer = Trainer(network,x_train,t_train,x_val,t_val,
                      epochs = epocs,mini_batch_size = 100,
                      optimizer = 'sgd',optimizer_param={'lr':lr},verbose = False)
    trainer.train()

    return trainer.test_acc_list,trainer.train_acc_list

#超参数搜索
optimization_trial = 100#实验100次超参数
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    weight_decay = 10**np.random.uniform(-8,-4)
    lr = 10**np.random.uniform(-6,-2)

    val_acc_list,train_acc_list = __train(lr,weight_decay)
    print('val acc:' + str(val_acc_list[-1]) + ' | lr:' + str(lr) + ', weight_decay:' + str(weight_decay))
    key = 'lr:' + str(lr) + ', weight decay:' + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

#绘制图形
print('=========== Hyper-Parameter Optimization Result ===========')
graph_draw_num = 20
col_num = 5
#ecil向上取整但是还是返回浮点数,使用int强制整数
row_num = int(np.ceil(graph_draw_num/col_num))
i = 0

#sorted是排序函数
    #.item()每次返回一个元组抱着分开的键和值,可迭代对象本质上是一个参数,相当于一个容器
    #sorted函数中的key是自己的参数,专门接收一个lambda函数提取值,然后按照这个值进行排序
    #lambda函数中的x对应取出来的元组.[1]取出来值 [-1]取出来值中的最后一个值
#key可以生成代表值,代表值与原元素有相应的链接关系
    #此时虽然第一个参数是迭代器,但是原元素就是字典中的键值对
    #所以可以根据代表值们的大小找到原元素,按照代表值升序排列后,对应的原元素也根据绑定的排序后的代表值按照同样的顺序排序
for key,val_acc_list in sorted(results_val.items(),key = lambda x:x[1][-1],reverse = True):
    print('Best-' + str(i+1) + '(val acc:' + str(val_acc_list[-1]) + ') | ' + key)

    plt.subplot(row_num,col_num,i+1)
    plt.title('Best-' + str(i+1))
    plt.ylim(0.0,1.0)
    #如果不是第一列就隐藏y刻度
    if i % 5:plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    #绘制验证数据图像
    plt.plot(x,val_acc_list)
    #绘制训练数据图像
    plt.plot(x,results_train[key],'--')
    i += 1

    if i>=graph_draw_num:
        break

plt.show()