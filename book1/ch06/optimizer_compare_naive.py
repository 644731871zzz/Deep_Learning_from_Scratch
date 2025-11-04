import sys, os
current_script_path = os.path.abspath(__file__)#
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common.optimizer import *
import numpy as np
import matplotlib.pyplot as plt
#来自python自带的语法库,现在字典已经自带顺序,不需要了
from collections import OrderedDict


def f(x,y):
    return x**2 / 20.0 + y**2

def df(x,y):
    return x / 10.0 , 2.0*y

init_pos = (-7.0,2.0)
params = {}
#仅仅展示
params['x'],params['y'] = init_pos[0],init_pos[1]#
grads = {}
grads['x'],grads['y'] = 0,0

optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr = 0.95)
optimizers['Momentum'] = Momentum(lr = 0.1)
optimizers['AdaGrad'] = AdaGrad(lr = 1.5)
optimizers['Adam'] = Adam(lr = 0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'],params['y'] = init_pos[0],init_pos[1]

    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])

        grads['x'],grads['y'] = df(params['x'],params['y'])
        optimizer.update(params,grads)

    #固定间隔序列
    x = np.arange(-10,10,0.01)#
    y = np.arange(-5,5,0.01)

    X,Y = np.meshgrid(x,y)
    Z = f(X,Y)

    #生成同形状布尔数组
    mask = Z > 7
    #布尔索引区别于切片操作(返回视图) 这里返回的是一个不同内存的副本
        #这是一种高级索引,高级索引包括布尔索引,正数索引,花式索引
        #如果是布尔值的数组,会自动进行布尔索引
        #a = Z[mask]布尔索引返回布尔数组为1的对应Z的位置的值组成的1D数组 - 这是高级索引
    #如果是一下赋值操作,会直接在布尔数组为True的对应Z的位置赋值0 - 这是赋值操作
    #高级索引和赋值操作是两个操作
        #等号在哪一侧会体现出不同的操作
        #单独调用Z[mask]会当成在等号右侧的高级索引取值,返回Z的对应索引提取出来的1D值
            #在等号右侧先进行取值再赋值 单独调用就是取值
        # Z[mask] =  这是在等号左侧,按照mask掩码就地写回
            #在等号左侧就是根据掩码将等号右侧值赋值
    Z[mask] = 0

    #是语法糖,绘制2x2网格的第idx个子图
    #如果没有fig自动创建绘制,如果有根据idx继续添加子图
    #语法糖的idx是从1开始循环 如果超过2x2给出的4 5,6,7,8...会依次按照顺序从索引为1的位置重新绘图
    plt.subplot(2,2,idx)
    idx += 1
    plt.plot(x_history,y_history,'--',color = 'red')
    plt.contour(X,Y,Z)
    plt.ylim(-10,10)
    plt.xlim(-10,10)
    plt.plot(0,0,'+')
    plt.title(key)
    plt.xlabel('x')
    plt.ylabel('y')

plt.show()