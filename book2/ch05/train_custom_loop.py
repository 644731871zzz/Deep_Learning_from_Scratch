import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm


batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5 #时间跨度大小
lr = 0.1
max_epoch = 100

corpus,word_to_id,id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]
ts = corpus[1:]
data_size = len(xs)
print('corpus size: %d,vocabulary size: %d' % (corpus_size,vocab_size))

# //是整除,只保留整数 (batch_size * time_size)这是一次训练消耗的数据量 
#   得到需要多少次训练才能遍历完成整个语料库
max_iters = data_size // (batch_size * time_size)
time_idx = 0 #T的位置
total_loss = 0 #累计loss
loss_count = 0 #loss计算次数
ppl_list = [] #困惑度列表

model = SimpleRnnlm(vocab_size,wordvec_size,hidden_size)
optimizer = SGD(lr)

#一簇多少个
jump = (corpus_size - 1) // batch_size
#每一簇的起始位置
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        batch_x = np.empty((batch_size,time_size),dtype = 'i')
        batch_t = np.empty((batch_size,time_size),dtype = 'i')
        for t in range(time_size):
            for i,offset in enumerate(offsets):
                #只有在offset + time_idx大于data_size(就是xs的长度)时候才取余数
                    #余数读取开头的xs作为输入
                batch_x[i,t] = xs[(offset + time_idx) % data_size]
                batch_t[i,t] = ts[(offset + time_idx) % data_size]

            time_idx += 1

        loss = model.forward(batch_x,batch_t)
        model.backward()
        optimizer.update(model.params,model.grads)
        total_loss += loss
        loss_count += 1

    #model.reset_state() #每个纪元不需初始化h,这里只有一篇文章,简单处理了
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch + 1,ppl))
    ppl_list.append(float(ppl))
    total_loss,loss_count = 0,0

x = np.arange(len(ppl_list))
plt.plot(x,ppl_list,label = 'train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()