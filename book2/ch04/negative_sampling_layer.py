import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.np import *
from common.layers import Embedding,SigmoidWithLoss
import collections

class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W) #这里的W没用理论矩阵乘法的形状,和Win的形状一样
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None #存正向传播的数据

    def forward(self,h,idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W*h,axis = 1) #numpy是对应元素相乘,再相加后会降维
        self.cache = (h,target_W)
        return out
    
    def backward(self,dout):
        h,target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W#
        return dh
    

class UnigramSampler:
    def __init__(self,corpus,power,sample_size):
        """
        corpus是语料库单词id
        power是对词做平滑的参数
        sample_size是每次采样几个样本
        """
        self.sample_size = sample_size#
        self.vocab_size = None #词汇表大小 占位
        self.word_p = None #每个词的概率 占位

        #计数器(count)
        #输出是字典,输入时候标记上,自动计数.不存在的词频,默认是0
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)#初始化word_p
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        #直接对词频处理即可
        self.word_p = np.power(self.word_p,power)
        self.word_p /= np.sum(self.word_p) #得到优化的归一化词频概率

    def get_negative_sample(self,target):
        #target是一组正例词汇,计算数量
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size,self.sample_size),
                                       type = np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i,:] = np.random.choice(self.vocab_size,
                                                        size = self.sample_size,
                                                        replace = False,
                                                        p = p)
        else:
            negative_sample = np.random.choice(self.vocab_size,
                                               size = (batch_size,self.sample_size),
                                               replace=True,
                                               p = self.word_p)
        return negative_sample