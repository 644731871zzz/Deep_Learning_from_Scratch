import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common import config
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity,to_gpu
from dataset import ptb
from better_rnnlm import BetterRnnlm



wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5

corpus,word_to_id,id_to_word = ptb.load_data('train')
#直接是word对应的id, 是验证集.因为验证集参与了模型选择,所以还需要测试数据
    #虽然模型选择没有参与训练
corpus_val,_,_ = ptb.load_data('val')
corpus_test,_,_ = ptb.load_data('test')

if config.GPU:
    corpus = to_gpu(corpus)
    corpus_val = to_gpu(corpus_val)
    corpus_test = to_gpu(corpus_test)

vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

model = BetterRnnlm(vocab_size,wordvec_size,hidden_size,dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model,optimizer)

best_ppl = float('inf') #字符串生成特殊浮点数
for epoch in range(max_epoch):
    trainer.fit(xs,ts,max_epoch = 1,
                time_size = time_size,max_grad=max_grad)
    
    model.reset_state()
    #corpus_val是向量的形式,eval_perplexity接收向量形式
    ppl = eval_perplexity(model,corpus_val)
    print('valid perplexity:',ppl)

    if best_ppl > ppl:
        best_ppl = ppl
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr

    model.reset_state()#清楚eval_perplexity留下来的隐藏状态
    print('-' * 50)

model.reset_state()
ppl_test = eval_perplexity(model,corpus_test)
print('test perplexity:',ppl_test)