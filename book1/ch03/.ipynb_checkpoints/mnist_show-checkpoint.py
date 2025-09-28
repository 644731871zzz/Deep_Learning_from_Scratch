import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    #长和np的array数组使用  np.uint8表示数据类型,无符号8位整数(0-255)
    #是类(模块)下的一个函数,输出Image格式对象
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

#在图像集中axis = 0时样本数(这是人为规定,区别于numpy的数组)然后axis = 1,2才是行列
(x_train,t_train),(x_text,t_test) = load_mnist(flatten = True,#单张图的维度变为1D
                                               normalize = False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)