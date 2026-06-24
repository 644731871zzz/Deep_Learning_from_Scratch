import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    #sample_size表示采样单词的数量(生成数量)
    #skip_ids可以给id列表,将不会采样 - 一些被预处理为统一标识的稀有单词,数字...
    def generate(self,start_id,skip_ids = None,sample_size = 100): 
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size: #只要word_ids数量不到就继续生成
            x = np.array(x).reshape(1,1)
            score = self.predict(x) #x必须是二维数组,因为输入是mini_batch的格式
            p = softmax(score.flatten())

            #这个len(p)输入后,默认输入arange(0,len(p))的等价的那种数组
            #第一个参数输入的是数组,需要和p的大小一致
            sampled = np.random.choice(len(p),size = 1,p = p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                x = np.squeeze(x)#去除维度
                word_ids.append(int(x))

        return word_ids
    

class RetterRnnlmGen(BetterRnnlm):
    def generate(self,start_id,skip_ids = None,sample_size = 100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1,1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p),size = 1,p = p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                x = np.squeeze(x)
                word_ids.append(int(x))

        return word_ids
    
    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h,layer.c))
        return states