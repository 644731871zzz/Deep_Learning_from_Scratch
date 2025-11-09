import os,sys
#abolute path 绝对路径,写出__file__文件的完整的路径 __file__是内置变量,表示当前脚本文件目录
    #前后带有'__'表示python自带的魔术方法或内置属性
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve#
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *

(x_train,t_train),(x_test,t_test) = load_mnist(normalize = True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

optimizers = {}
#类带有()表示已经创建了一个类实例,已经调用了__init__
#self永远是字典当前创建出来的对象,然后让字典的'SGD'指向这个对象
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}

for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size = 784,hidden_size_list = [100,100,100,100],
        output_size = 10
    )#
    train_loss[key] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size,batch_size) #有放回抽样,可能重复抽取,代码照常执行
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch,t_batch)
        optimizers[key].update(networks[key].params,grads)

        loss = networks[key].loss(x_batch,t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0 :
        print( '===========' + 'iteration:' + str(i) + '===========')
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch,t_batch)
            print(key + ':' + str(loss))

markers = {'SGD':'o','Momentum':'x','AdaGrad':'s','Adam':'D'}
x = np.arange(max_iterations)
for key in optimizers.keys():
    #取值范围自动取值输入x的最小值到最大值,y同理
    #smooth_curve对一维数据点的波动平滑处理
    #markevery 表示间隔多少点画一个
    plt.plot(x,smooth_curve(train_loss[key]),marker = markers[key],markevery = 100,label = key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0,1)
plt.legend()
plt.show()
