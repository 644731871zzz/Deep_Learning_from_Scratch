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
        self.sample_size = sample_size#负例词大小
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
                                       dtype = np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum() #处理不带当前词汇的词频
                negative_sample[i,:] = np.random.choice(self.vocab_size, #从这里抽
                                                        size = self.sample_size,#抽多少个
                                                        replace = False,#不放回
                                                        p = p)#对应概率
        else:
            negative_sample = np.random.choice(self.vocab_size,
                                               size = (batch_size,self.sample_size),
                                               replace=True,
                                               p = self.word_p)
        return negative_sample
    

class NegativeSamplingLoss:
    def __init__(self,W,corpus,power = 0.75,sample_size = 5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus,power,sample_size)
        #+1是因为正例的层数多一个
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params,self.grads = [],[]
        for layer in self.embed_dot_layers:
            #这里最后定位到的是cbow里面创建的最原始的W_in
            #numpy只要出现+= -=这种类似的语法,一定会修改原始定位到的数据
            self.params += layer.params
            self.grads += layer.grads

    #假设layers的第0个用于正例
    def forward(self,h,target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        #正例正向传播
        score = self.embed_dot_layers[0].forward(h,target)
        correct_label = np.ones(batch_size,dtype = np.int32)
        loss = self.loss_layers[0].forward(score,correct_label)

        #负例的正向传播
        negative_label = np.zeros(batch_size,dtype= np.int32)
        for i in range(self.sample_size):
            #取出 batch 中每个样本的第 1 个负例
            negative_target = negative_sample[:,i]
            #这里的h和么个样本的第一个负例的大小对应
            score = self.embed_dot_layers[1+i].forward(h,negative_target)
            loss += self.loss_layers[1+i].forward(score,negative_label)

        return loss
    

    def backward(self,dout = 1):
        dh = 0
        for l0,l1 in zip(self.loss_layers,self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh