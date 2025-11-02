import sys,os
sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict #

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        #初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层(仿射层+激活函数)
        #有序字典,可记住添加到字典中的元素的顺序
            #新版python中普通字典也可以插入顺序了
        self.layers = OrderedDict()
        #这种针对键值对的值赋值,同键赋值覆盖原条目
            #使用{}会创建新字典,所以这是特有的键值对语法
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        #Relu激活函数
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        #单独给一个名称给输出函数和损失函数的整合
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        #.values()为取值不取键,无参数取所有值,按照顺序字典的顺序返回
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis = 1)
        #条件为真就执行后面的语句,否则什么都不做(不报错)
            #这里不报错else可省略,即使是正常的if不写else为否也会跳过 else和elif为补充
        if t.ndim != 1 : t = np.argmax(t,axis = 1) #

        accuracy = np.sum(y == t)/float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self,x,t):
        loss_W = lambda W : self.loss(x,t)

        grads = {}
        #注意这里不是用类的这里的命名函数,而是from中的一个函数计算,如果用类中的用self.numerical...了
        grads['W1'] = numerical_gradient(loss_W,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])

        return grads
    
    def gradient(self,x,t):
        #forward
        #先正向传播,各层存储所需变量反向传播用
        self.loss(x,t)

        #backward
        dout = 1
        #输出函数和损失函数层的反向传播
        dout = self.lastLayer.backward(dout)

        #list是创建个新的[]对象,里面放上这些东西
            #省略了for循环使用.append逐一添加
        layers = list(self.layers.values())
        #反转列表得到反向传播的顺序
            #列表使用索引定位,字典使用键值定位,最后都指向了那个类,所以类没变换,没动字典,字典也没变化
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #设定
        grads = {}
        grads['W1'],grads['b1'] = self.layers['Affine1'].dW,self.layers['Affine1'].db
        grads['W2'],grads['b2'] = self.layers['Affine2'].dW,self.layers['Affine2'].db

        return grads