import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
# 返回训练/测试数据的ID序列，以及字符和ID的双向转换字典
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from attention_seq2seq import AttentionSeq2seq

(x_train,t_train),(x_test,t_test) = sequence.load_data('date.txt')
char_to_id,id_to_char = sequence.get_vocab()

x_train,x_test = x_train[:,::-1],x_test[:,::-1] #反向输入

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 256
batch_size = 128
max_epoch = 10
max_grad = 5.0

model = AttentionSeq2seq(vocab_size,wordvec_size,hidden_size)


optimizer = Adam()
trainer = Trainer(model,optimizer)

acc_list = [] #

for epoch in range(max_epoch):
    trainer.fit(x_train,t_train,max_epoch = 1,
                batch_size = batch_size,max_grad= max_grad)
    
    correct_num = 0
    for i in range(len(x_test)):
        question,correct = x_test[[i]],t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model,question,correct,
                                    id_to_char,verbose,is_reverse= True)
        
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' %(acc * 100)) #.3f后面的%%表示打印一个%作为显示

model.save_params()