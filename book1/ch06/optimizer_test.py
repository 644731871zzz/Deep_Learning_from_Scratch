import numpy as np

class Momentum:
    def __init_(self,lr = 0.01,momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key,val in params.iterms():
                #创建匹配形状,起始速度为0
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self,lr = 0.01):
        self.lr = lr
        self.h = None

    def update(self,params,grads):
        #is是比较对象身份 是否是一个对象(内存位置是否相同)
            #==是比较值是否相等
        if self.h is None:
            self.h = {}
            for key,val in params.item():
                self.h[kay] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            #最后的1e-7是防止除数有0
            params[key] -=self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
