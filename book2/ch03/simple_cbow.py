import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.layers import MatMul,SoftmaxWithLoss

import numpy as np


class SimpleCBOW:
    #vocab_size表示词汇总数
    def __init__(self,vocab_size,hidden_size):
        V,H = vocab_size,hidden_size
        
        W_in = 0.01*np.random.randn(V,H).astype('f') #32为浮点数
        W_out = 0.01*np.random.randn(H,V).astype('f')

        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        #整理所有的权重梯度
        layers = [self.in_layer0,self.in_layer1,self.out_layer]
        #梯度已经在初始化时候就确定了位置,反向传播才有真的梯度
        self.params,self.grads = [],[]
        for layer in layers:
            #+=用在python数列是当前维度的向后拼接,python没有维度,所以只看这一层装了什么
                #元组也可以,但是不能修改元组,其实是得到了新的元组
                #numpy数组用+=变成了当前维度形状下对应位置的的加法
                #.shape看的是内部的,所以忽略了numpy的'壳'
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self,contexts,target):
        #contexts假定已经将上下文转化为了3维数组 ,target依旧二维
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0 + h1) * 0.5

        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score,target)
        return loss
    
    def backward(self,dout = 1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None