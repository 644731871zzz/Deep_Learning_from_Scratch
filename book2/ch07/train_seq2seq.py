import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq #评估模型
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


(x_train,t_train),(x_test,t_test) = sequence.load_data('addition.txt')
char_to_id,id_to_char = sequence.get_vocab()#load_data后,两个字典已经加载

#翻转输入
is_reverse = True
if is_reverse:
    x_train,x_test = x_train[:,::-1],x_test[:,::-1] #[::-1] start:end:step

vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

#model = Seq2seq(vocab_size,wordvec_size,hidden_size)
model = PeekySeq2seq(vocab_size,wordvec_size,hidden_size)

optimizer = Adam()
trainer = Trainer(model,optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train,t_train,max_epoch = 1,
                batch_size = batch_size,max_grad = max_grad)
    
    correct_num = 0
    for i in range(len(x_test)):
        #用[[i]]嵌套不论x_test的数组维度如何,取最外围i的数据后,保持初始维度
        question,correct = x_test[[i]],t_test[[i]] 
        verbose = i <10
        #评估模型
        #verbose是是否打印详细测试结果,如果i<10打印结果
        #每次输出0或1,对应错误和正确
        correct_num += eval_seq2seq(model,question,correct,
                                    id_to_char,verbose)
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc*100))

x = np.arange(len(acc_list))
plt.plot(x,acc_list,marker = 'o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1.0)
plt.show()