import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import os
import pickle
from common.np import *
from common.util import to_gpu,to_cpu

class BaseModel:
    def __init__(self):
        self.params,self.grads = None,None#占位

    def forward(self,*args):#接收任意大小的参数
        raise NotImplementedError#提示是未实现的错误,因为后面继承时候的子类会实现
    
    def backward(self,*args):
        raise NotImplementedError 
    
    def save_params(self,file_name = None):
        if file_name is None:
            #如果不传入文件名,自己生成文件名
            #用类名作为文件名
            #__表示特殊属性,不是隐式调用,这里调用出来的名字格式是字符串格式
            file_name = self.__class__.__name__ + '.pkl'

        #以16位浮点数存储
        params = [p.astype(np.float16) for p in self.params]

        if GPU:
            #如果数据在GPU上,这里将数据转为cpu,因为数据都以cpu的格式存储
            params = [to_cpu(p) for p in params]

        with open(file_name,'wb') as f:
            pickle.dump(params,f)

    def load_params(self,file_name = None):
        if file_name is None:
            file_name = self.__class__.__name__ + '.pkl'

        if '/' in file_name:
            #将'/'替换为系统的分隔符
                #防止系统不一样 win是\ mac和linux是 /
                #返回系统的文件夹分隔符的字符串形式
            file_name = file_name.replace('/',os.sep)
        
        #如果文件不存在,报错,报错返回的是布尔值
        if not os.path.exists(file_name):
            #创建一个里面带信息的报错对象,输入为字符串
            raise IOError('No file:' + file_name)
        
        with open(file_name,'rb') as f:
            params = pickle.load(f)

        params = [p.astype('f') for p in params]
        if GPU:
            #gpu模式运算按照gpu模式读取
            params = [to_gpu(p) for p in params]
        for i,param in enumerate(self.params):
            #替换param本身里面的所有的值
                #防止param指向其他值,这里param本身不变
            param[...] = params[i] 