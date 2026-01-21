import sys
#.表示当前目录,..表示父目录
sys.path.append('..')
from common.np import *

class SGC:
    def __init__(self,lr = 0.01):
        self.lr = lr

    def update(self,params,grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]#只取最外层索引,最外层索引下就是W和grads,形状对应不是1D数组