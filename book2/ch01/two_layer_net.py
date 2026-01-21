import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from common.layers import Affine,Sigmoid,SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size

        W1 = 0.01*np.random.randn(I,H)
        b1 = np.zeros(H)
        W2 = 0.01*np.random.randn(H,O)
        b2 = np.zeros(O)

        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params,self.grads = [],[]

        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads #提前的一个内存引用,因为共享内存,内存在反向传播时候改变了内存,但是引用没变导致直接能够读取到

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)

        return x
    
    def forward(self,x,t):
        score = self.predict(x)
        loss = self.losslayer.forward(score,t)
        return loss
    
    def backward(self,dout = 1):
        dout = self.loss_layer.backward
        #reversed将迭代对象迭代顺序改为反向
            #部分迭代对象可用,一般的内置序列类型全部可用,包括了元组,列表,字符串,range()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout#基本没用,都存在grad中了