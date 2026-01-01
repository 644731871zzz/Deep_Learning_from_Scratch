import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

def filter_show(filters,nx = 8,margin = 3,scale = 10):#nx表示每一行显示多少个滤波器,x表示x方向.最后两个分别表示子图间隔和单个滤波器放大倍数
    FN,C,FH,FW = filters.shape#数量,通道数量,高度,宽度
    ny = int(np.ceil(FN/nx))#ceil向上取整,计算列数量,y表示y的方向

    fig = plt.figure()
    #fig的调整参数,前四个参数配置了子图占整个画面的比例,后两个配置子图之间了垂直和水平间距
    #这四个比例表示占满整个画布
    fig.subplots_adjust(left = 0,right = 1,bottom = 0,top = 1,hspace = 0.05,wspace = 0.05)

    #逐一选出每一个滤波器
    for i in range(FN):
        ax = fig.add_subplot(ny,nx,i+1,xticks = [],yticks = [])
        #[i,0]等价于[i,0,:,:]少写的按照:补全
        ax.imshow(filters[i,0],cmap = plt.cm.gray_r,interpolation = 'nearest')#最后一个参数表示不做插值平滑(最临近插值)

    plt.show()

#初始化并显示权重
network = SimpleConvNet()
filter_show(network.params['W1'])

#读取学习后的权重并显示
network.load_params('params.pkl')
filter_show(network.params['W1'])