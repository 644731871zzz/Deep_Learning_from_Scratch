import numpy as np

class Sigmoid:
    def __init__(self):
        #这个层不需要参数,用空列表表示初始化参数
        self.params = []

    def forward(self,x):
        return 1/(1+np.exp(-x))
    

class Affine:
    def __init__(self,W,b):
        self.params = [W,b]

    def forward(self,x):
        W,b = self.params
        out = np.dot(x,W) + b
        return out
    

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size):
        I,H,O = input_size,hidden_size,output_size

        #初始化权重和偏置
        W1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        #生成层
        self.layers = [
            Affine(W1,b1),
            Sigmoid(),
            Affine(W2,b2)
        ]

        self.params = []
        for layer in self.layers:
            #区别于append,append直接将所有的当前的表示添加到最后,类似[W,b]直接添加到最后.[[W,b],[]...]
            #如果+=后面是一个迭代器,使用+=会将参数拆开添加 这个+=仅仅拆开一层
                #所以结果可以为[W,b,[]]输入的[W,b]被拆开
                #这是容器的语法,不是变量语义,表示对list做原地扩展.将+=后面的元素当做一个可迭代序列,将序列中的元素逐一添加(不是一起)
                    #类似拼接操作
            self.params += layer.params

    def predict(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    

#实际推理
x = np.random.randn(10,2)
model = TwoLayerNet(2,4,3)
s = model.predict(x)
print(s)