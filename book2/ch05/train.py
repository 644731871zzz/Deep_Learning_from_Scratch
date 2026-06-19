import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.optimizer import SGD
from common.trainer import RnnlmTrainer #类似神经网络的Trainer类,这里的专门用在RNN里面
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 5
lr = 0.1
max_epoch = 100

corpus,word_to_id,id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]
ts = corpus[1:]

model = SimpleRnnlm(vocab_size,wordvec_size,hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model,optimizer)

trainer.fit(xs,ts,max_epoch,batch_size,time_size) #学习
trainer.plot()