import os,sys
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus,word_to_id,id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
param_path = os.path.join(parent_dir, 'ch06', 'Rnnlm.pkl')
model.load_params(param_path)

start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N','<unk>','$'] #数字,低频单词,美元符号
skip_ids = [word_to_id[w] for w in skip_words]

word_ids = model.generate(start_id,skip_ids)
#join将前面字符用作分隔符,分隔后面的列表.这是字符串的方法
#join接受的列表里的元素必须都是字符串
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace('<eos>','.\n')#句子结束符替换为换行
print(txt)