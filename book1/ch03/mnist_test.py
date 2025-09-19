#sys指的是python的解释其相关 os表示操作系统
import sys,os
#pardir 是 parent directory 父目录
#这里将父目录加入模块搜索路径
#sys.path表示一个目录列表,import会去这里寻找模块和包
    #append把一个目录追加进去
        #os.pardir将父目录添加到搜索路径
sys.path.append(os.pardir)
#将mnist模块导入 .py文件就是模块 文件夹是包,这里dataset就是一个包 若干包和模块组成库
    #load_mnist是模块中的函数
#import导入模块,包,库都可以.
from dataset.mnist import load_mnist

#训练图像,训练标签,测试图像,测试标签
#train表示训练的意思
(x_train,t_train),(x_test,t_test) = load_mnist(flatten = True,#是否展开图像为一维数组 如果为False将为1x28x28三维数组
                                               normalize = False) #是否归一化(正规化) 为False表示保持原来的0-255

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
