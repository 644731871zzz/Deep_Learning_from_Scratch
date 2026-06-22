import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 4
max_grad = 0.25 #进行梯度裁剪

corpus,word_to_id,id_to_word = ptb.load_data('train')
corpus_test,_,_ = ptb.load_data('test')#
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = Rnnlm(vocab_size,wordvec_size,hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model,optimizer)


#fit里面可以梳理梯度爆炸.是先在 backward 里传播完，梯度h已经变大后，再做梯度裁剪。
#eval_interval表示每20个纪元显示一次困惑度
trainer.fit(xs,ts,max_epoch,batch_size,time_size,max_grad,eval_interval = 20)
trainer.plot(ylim = (0,500))

model.reset_state()#重置h和c(隐藏状态和记忆单元)
#用测试集评估困惑度
ppl_test = eval_perplexity(model,corpus_test)
print('test perplexity:' , ppl_test)

model.save_params()