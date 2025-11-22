import os,sys
#abolute path 绝对路径,写出__file__文件的完整的路径 __file__是内置变量,表示当前脚本文件目录
    #前后带有'__'表示python自带的魔术方法或内置属性
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from common.optimizer import *

class Trainer:
    """进行神经网络训练的类"""
    def __init__(self,network,x_train,t_train,x_test,t_test,
                 epochs = 20,mini_batch_size = 100,
                 optimizer = 'SGD',optimizer_param = {'lr':0.01},
                 #用多少样本数进行评估,是否带音训练过程日志
                 #如果第一个参数是None,则用全部数据集进行评估
                 evaluate_sample_num_per_epoch = None,verbose = True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch#

        optimizer_class_dict = {'sgd':SGD,
                                'momentum':Momentum,
                                'nesterov':Nesterov,
                                'adagrad':AdaGrad,
                                'rmsprpo':RMSprop,
                                'adam':Adam}
        #**用于吧字典展开成关键字参数  'a':0.1 >>> a = 0.1
            #如果平时这样用就变成了一次变量赋值,在函数输入参数中可以用作关键字参数
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size,1)#一个纪元迭代多少次
        self.max_iter = int(epochs*self.iter_per_epoch)#所有纪元下,一共训练多少次(多少个batch)
        self.current_iter = 0
        self.current_epoch = 0

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []


    def train_step(self):
        batch_mask = np.random.choice(self.train_size,self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch,t_batch)
        self.optimizer.update(self.network.params,grads)
        
        loss = self.network.gradient(x_batch,t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print('train loss:' + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:
            self.curren_epoch += 1

            x_train_sample,t_train_sample = self.x_train,self.t_train
            x_test_sample,t_test_sample = self.x_test,self.t_test
            if not self.evaluate_sample_num_per_epoch in None:
                t = self.evaluate_sample_num_per_epoch#为了简化没有取随机,只是取t值前面的进行快速处理
                x_train_sample,t_train_sample = self.x_train[:t],self.t_train[:t]
                x_test_sample,t_test_sample = self.x_test[:t],self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample,t_train_sample)
            test_acc = self.network.accuracy(x_test_sample,t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose: print ('=== epoch:' + str(self.current_epoch) + ',train acc:' + str(train_acc) + ',test acc:' + str(test_acc) + '===')
        self.current_iter += 1

        def train(self):
            for i in range(self.max_iter):
                self.train_step()#用前面的函数完成所有训练,次数限制也在这里,这里

            test_acc = self.network.accuracy(self.x_test,self.t_test)

            if self.verbose:
                print('=========== Final Test Accuracy ===========')
                print('test acc' + str(test_acc))